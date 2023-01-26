import functools
from io import BytesIO
from os.path import join
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image
from sqlalchemy.engine import Row

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


@functools.lru_cache
def get_dummy_image_bytes() -> bytes:
    memory_file = BytesIO()
    image = Image.new("RGB", (1, 1))
    image.save(memory_file, "JPEG")
    return memory_file.getvalue()


def prepare_batch(result_batch: Iterator[Sequence[Row]]) -> Optional[pd.DataFrame]:
    """
    Get and return images for all rows that have non-NaN "filepath" entries, None if there aren't any.
    :param result_batch:
    :return:
    """
    batch_df = pd.DataFrame(result_batch)
    batch_df["image_bytes"] = fetch_images_from_lmdb(batch_df["lmdb_key"])

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
        index_bytes = faiss.serialize_index(index)
        with LmdbWriter(LMDB_PATH) as writer:
            writer.add(grid.encode(), index_bytes)
