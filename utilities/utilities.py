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
from sqlalchemy import create_engine

from utilities.lmdb_classes import LmdbReader, LmdbWriter

"""Various global variables used in the inference process"""
BATCH_SIZE = 32
KP_ID_MODEL_INPUT_IMAGE_SIZE = (224, 224)
ROT_MODEL_INPUT_IMAGE_SIZE = (128, 128)
FILES_PATH = "files"
LMDB_PATH = "lmdb"
ZIP_NAMES = [
    "whareorino_a.zip",
    "whareorino_b.zip",
    "whareorino_c.zip",
    "whareorino_d.zip",
    "pukeokahu.zip",
]
SQL_SERVER_STRING = "postgresql://postgres:PepeketuaFrogs@sql-server/postgres"
WHAREORINO_EXCEL_FILE = join(FILES_PATH, "whareorino.xls")
PUKEOKAHU_EXCEL_FILE = join(FILES_PATH, "pukeokahu.xls")
GRID_NAMES = ["Grid A", "Grid B", "Grid C", "Grid D", "Pukeokahu Frog Monitoring"]
LANDMARK_MODEL = join(FILES_PATH, "landmark_model_714")
ROTATION_MODEL = join(FILES_PATH, "rotation_model_weights_10")
IDENTIFY_MODEL = join(FILES_PATH, "ep29_vloss0.0249931520129752_emb.ckpt")
EMBEDDING_LENGTH = 64
DEFAULT_K_NEAREST = 10

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
) -> Image.Image:
    try:
        image = jpeg.decode(
            image_bytes, turbojpeg.TJPF_RGB, scaling_factor=scaling_factor
        )
        if image.shape[2] != 3:
            raise ValueError
        else:
            return image
    except ValueError:
        logger.warning(
            "get_image() tried opening an image that isn't RGB. Trying to force it to be RGB..."
        )
        image = force_image_to_rgb(Image.open(BytesIO(image_bytes)))
        return image


@st.experimental_memo
def get_row_and_image_by_id(id: Union[int, float]) -> Tuple[pd.DataFrame, bytes]:
    with SqlQuery() as (connection, frogs):
        statement = db.select(frogs).where(frogs.c.id == id)
        result = connection.execute(statement)
        row = pd.DataFrame(result.fetchall())

    try:
        with LmdbReader(LMDB_PATH) as reader:
            image_bytes = reader.read(row.at[0, "lmdb_key"].encode())
    except Exception as e:
        logger.error(f"Error fetching id {id} from Postgres or LMDB, id not found!")
        raise e

    return row, image_bytes


@st.experimental_singleton
def get_usage_instructions():
    with open("utilities/USAGE.md", "r") as file:
        return file.read()
