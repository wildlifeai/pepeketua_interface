from io import BytesIO
from os.path import join
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import sqlalchemy as db
import streamlit as st
from loguru import logger
from PIL import Image
from sqlalchemy import create_engine

from utilities.lmdb_classes import LmdbReader, LmdbWriter

"""Various global variables used in the inference process"""
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
ROT_IMAGE_SIZE = (128, 128)
PHOTO_PATH = "pepeketua_id"
LMDB_PATH = join(PHOTO_PATH, "lmdb")
ZIP_NAMES = [
    "whareorino_a.zip",
    "whareorino_b.zip",
    "whareorino_c.zip",
    "whareorino_d.zip",
    "pukeokahu.zip",
]
SQL_SERVER_STRING = "postgresql://lioruzan:nyudEce5@localhost/frogs"
WHAREORINO_EXCEL_FILE = join(
    PHOTO_PATH,
    "Whareorino frog monitoring data 2005 onwards CURRENT FILE - DOCDM-106978.xls",
)
PUKEOKAHU_EXCEL_FILE = join(
    PHOTO_PATH, "Pukeokahu Monitoring Data 2006 onwards - DOCDM-95563.xls"
)
GRID_NAMES = ["Grid A", "Grid B", "Grid C", "Grid D", "Pukeokahu Frog Monitoring"]
LANDMARK_MODEL = "model_weights/landmark_model_714"
ROTATION_MODEL = "model_weights/rotation_model_weights_10"
IDENTIFY_MODEL = "model_weights/ep29_vloss0.0249931520129752_emb.ckpt"
EMBEDDING_LENGTH = 64
DEFAULT_K_NEAREST = 3


def fetch_images_from_lmdb(keys: pd.Series) -> List[bytes]:
    with LmdbReader(LMDB_PATH) as reader:
        image_list = [
            reader.read(key.encode()) if key is not None else None for key in keys
        ]
        return image_list


def get_image_size(image_bytes: bytes) -> Tuple:
    with Image.open(BytesIO(image_bytes)) as im:
        return im.width, im.height


def force_image_to_be_rgb(image: Image) -> Image:
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


def prepare_batch(batch_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Get and return images for all rows that have non-NaN "filepath" entries, None if there aren't any.
    :param batch_df: df containing all frog information, including lmdb_key.
                     Can contain existing image_bytes if images were loaded from an
                     external source (such as uploaded files)
    :return:
    """
    if "image_bytes" not in batch_df:
        batch_df.loc[:, "image_bytes"] = fetch_images_from_lmdb(batch_df["lmdb_key"])

    if bool(batch_df["image_bytes"].any()) is False:
        return None

    # Filter rows with no images
    batch_df = batch_df[batch_df["image_bytes"].notna()]

    # Populate size columns
    batch_df.loc[:, ["width_size", "height_size"]] = list(
        map(get_image_size, batch_df["image_bytes"])
    )
    # For later calculations
    batch_df.reset_index(drop=True, inplace=True)
    return batch_df


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


def save_indices_to_lmdb(indices: Dict[str, faiss.Index]):
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


def display_image_and_k_nn(row_num: int, image_bytes: bytes, k_nn: List[int]):
    st.write(
        f"# Displaying frog image {row_num} and it's {len(k_nn)} Nearest Neighbors"
    )
    with Image.open(BytesIO(image_bytes)) as image:
        st.image(image)

    rows, images = get_rows_and_images_from_ids(k_nn)
    st.image(images)

    # Close images
    list(map(lambda im: im.close(), images))


@st.experimental_memo
def get_rows_and_images_from_ids(ids: np.array) -> Tuple[pd.DataFrame, List["Image"]]:
    with SqlQuery() as (connection, frogs):
        statement = db.select(frogs).where(frogs.c.id.in_(ids))
        result = connection.execute(statement)
        df = pd.DataFrame(result.fetchall())

    with LmdbReader(LMDB_PATH) as reader:
        keys = [key.encode() for key in df["lmdb_key"].to_list()]
        images_bytes = reader.read_keys(keys)

    images = [Image.open(BytesIO(image_bytes)) for image_bytes in images_bytes]
    return df, images


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
