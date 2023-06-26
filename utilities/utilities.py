import pickle
import time
from io import BytesIO
from os.path import join
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
import sqlalchemy as db
import streamlit as st
import turbojpeg
from loguru import logger
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

from utilities.lmdb_classes import LmdbReader, LmdbWriter

"""Various global variables used in the inference process"""
BATCH_SIZE = 32
KP_ID_MODEL_INPUT_IMAGE_SIZE = (224, 224)
ROT_MODEL_INPUT_IMAGE_SIZE = (128, 128)
FILES_PATH = "pepeketua_files"
LMDB_PATH = "lmdb"
ZIP_NAMES = [
    "whareorino_a.zip",
    "whareorino_b.zip",
    "whareorino_c.zip",
    "whareorino_d.zip",
    "pukeokahu.zip",
]
SQL_SERVER_STRING = "postgresql://postgres:PepeketuaFrogs@sql-server/postgres"
#### UPDATE EXCEL NAMES ON NOTION ####
WHAREORINO_EXCEL_FILE = join(FILES_PATH, "whareorino.xls")
PUKEOKAHU_EXCEL_FILE = join(FILES_PATH, "pukeokahu.xls")
GRID_NAMES = ["Grid A", "Grid B", "Grid C", "Grid D", "Pukeokahu Frog Monitoring"]
LANDMARK_MODEL = join(FILES_PATH, "landmark_model_714")
ROTATION_MODEL = join(FILES_PATH, "rotation_model_weights_10")
IDENTIFY_MODEL = join(FILES_PATH, "identify_emb_model_weights_ep29")
EMBEDDING_LENGTH = 64
DEFAULT_KNN_VALUE = 10
MAX_NN_VALUE = 50


jpeg = turbojpeg.TurboJPEG()


class SqlQuery:
    """A simple class used to query our SQL server"""

    def __init__(self):
        self.engine = create_engine(SQL_SERVER_STRING)
        self.metadata = db.MetaData()

    def __enter__(self):
        self.connection = self.engine.connect()
        frogs = db.Table(
            "frogs", self.metadata, autoload=True, autoload_with=self.connection
        )
        return self.connection, frogs

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()
        self.engine.dispose()


def fetch_images_from_lmdb(keys: pd.Series) -> List[bytes]:
    with LmdbReader(LMDB_PATH) as reader:
        image_list = [
            reader.read(key.encode()) if key is not None else None for key in keys
        ]
        return image_list


@st.experimental_singleton
def fetch_scaler_from_lmdb() -> StandardScaler:
    with LmdbReader(LMDB_PATH) as reader:
        return pickle.loads(reader.read(b"scaler"))


def force_image_to_rgb(image: Image) -> Image:
    """Try to force image to be RGB, we don't support other modes"""
    if image.mode != "RGB":
        try:
            image = image.convert("RGB")
        except ValueError as error:
            logger.error(
                f"We don't support images that can't be converted to RGB such as {image}"
            )
            raise error
    return image


def time_it(func):
    """Simple decorator used to time functions and print their runtime"""

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.debug(f"{func.__name__} took {end - start:.6f} seconds")
        return result

    return wrapper


def to_float(x):
    """Used to convert unreliable columns to float"""
    try:
        return np.float64(x)
    except:
        return np.nan


class FeatureProcessing:
    NUMERIC_FEATURE_COLUMNS = ["SVL (mm)", "Weight (g)"]
    NUMERIC_FEATURE_AGG_DICT = {col: to_float for col in NUMERIC_FEATURE_COLUMNS}
    LOCAL_BINCODE_COLUMN = "Updated capture photo code"
    QUERY_BINCODE_COLUMN = "Capture photo code"
    INVALID_BINCODE_CHARACTERS = "_-?"
    FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + [QUERY_BINCODE_COLUMN]


def extract_features(rows: pd.DataFrame):
    features = (
        rows[FeatureProcessing.NUMERIC_FEATURE_COLUMNS]
        .agg(FeatureProcessing.NUMERIC_FEATURE_AGG_DICT)
        .to_numpy()
    )
    return features


def transform_numeric_features(features: np.array):
    scaler = fetch_scaler_from_lmdb()
    features = scaler.transform(features)
    return features


def extract_and_transform_numeric_features(rows: pd.DataFrame) -> np.array:
    features = extract_features(rows)
    features = transform_numeric_features(features)
    return features


def get_bincode_features(df: pd.DataFrame, is_query: bool) -> np.array:
    """Extract the binary coding given to identified frogs"""
    if is_query:
        bincode_column = FeatureProcessing.QUERY_BINCODE_COLUMN
    else:
        bincode_column = FeatureProcessing.LOCAL_BINCODE_COLUMN
    return df.loc[:, bincode_column].str.split("-", n=1, expand=True)[0].values


@np.vectorize
def compare_bincodes(a: Union[str, float], b: Union[str, float]) -> float:
    """
    Compare two frog binary codes and return their editing distance.
    The function is vectorized to allow comparing two arrays of codes with broadcasting.
    :param a: four letter binary code string (e.g. "0101")
    :param b: another four letter binary code string
    :return: normalized editing distance between the two codes
    """
    len_a = len(a)
    if len_a != len(b) or pd.isna(a) or pd.isna(b):
        return 1.0
    return sum(map(compare_letter, a, b)) / len_a


def compare_letter(a: str, b: str) -> float:
    """
    Compare two letters and return their editing distance.
    Since characters such as "_" means some joint wasn't identified, we return half a count for it.
    :param a:
    :param b:
    :return:
    """
    if (
        a in FeatureProcessing.INVALID_BINCODE_CHARACTERS
        or b in FeatureProcessing.INVALID_BINCODE_CHARACTERS
    ):
        return 0.5
    elif a != b:
        return 1.0
    else:
        return 0.0


def initialize_faiss_index():
    index = faiss.IndexFlatL2(EMBEDDING_LENGTH)
    index = faiss.IndexIDMap(index)
    return index


def initialize_faiss_indices():
    indices = dict()
    for grid in GRID_NAMES:
        indices[grid] = initialize_faiss_index()
    return indices


def update_indices(
    indices: Dict[str, faiss.Index],
    identity_vectors: np.array,
    ids: np.array,
    grids: pd.Series,
) -> Dict[str, faiss.Index]:
    grids.reset_index(drop=True, inplace=True)
    for grid in grids.unique():
        grid_indices = grids.index[grids == grid].to_numpy()
        grid_vectors = identity_vectors[grid_indices]
        grid_ids = ids[grid_indices]
        indices[grid].add_with_ids(grid_vectors, grid_ids)
    return indices


def save_faiss_indices_to_lmdb(indices: Dict[str, faiss.Index]):
    for grid, index in indices.items():
        index_bytes = faiss.serialize_index(index).tobytes()
        with LmdbWriter(LMDB_PATH) as writer:
            writer.add(grid.encode(), index_bytes)


@st.experimental_memo
def load_faiss_indices_from_lmdb() -> Dict[str, faiss.Index]:
    indices = dict()
    with LmdbReader(LMDB_PATH) as reader:
        for grid in GRID_NAMES:
            index_bytes = reader.read(grid.encode())
            index_array = np.frombuffer(index_bytes, dtype=np.uint8)
            indices[grid] = faiss.deserialize_index(index_array)
    return indices


@st.experimental_memo
def get_image(
    image_bytes: bytes, scaling_factor: Optional[Tuple[int, int]] = None
) -> np.array:
    image = jpeg.decode(image_bytes, turbojpeg.TJPF_RGB, scaling_factor=scaling_factor)
    if image.shape[2] == 3:
        return image
    else:
        # If the image is not RGB, we try to convert it to RGB
        image = force_image_to_rgb(Image.open(BytesIO(image_bytes)))
        return np.array(image)


def get_row_and_image_by_id(id: Union[int, float]) -> Tuple[pd.DataFrame, bytes]:
    row = get_capture_rows_by_ids([id])
    image_bytes = get_images_by_keys([row.at[0, "lmdb_key"]])[0]
    return row, image_bytes


@st.experimental_memo
def get_capture_rows_by_ids(ids: List[Union[int, float]]) -> pd.DataFrame:
    with SqlQuery() as (connection, frogs):
        statement = db.select(frogs).where(frogs.c.id.in_(ids))
        result = connection.execute(statement)
        rows = pd.DataFrame(result.fetchall())
        # Order the rows by the order of the ids, important for downstream processing
        rows = rows.set_index("id").loc[ids].reset_index(drop=True)
    return rows


@st.experimental_memo
def get_images_by_keys(keys: List[str]) -> List[bytes]:
    try:
        with LmdbReader(LMDB_PATH) as reader:
            image_list = [reader.read(key.encode()) for key in keys]
    except Exception as e:
        logger.error("Error fetching images from LMDB, a key was not found!")
        raise e
    return image_list


@st.experimental_singleton
def get_usage_instructions():
    with open("utilities/USAGE.md", "r") as file:
        return file.read()
