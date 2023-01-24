import functools
from io import BytesIO
from os.path import join
from typing import Iterator, List, Sequence, Tuple

import pandas as pd
from PIL import Image
from sqlalchemy.engine import Row

from utilities.lmdb_classes import LmdbReader

"""Various global variables used in the inference process"""
BATCH_SIZE = 3
IMAGE_SIZE = (224, 224)
ROT_IMAGE_SIZE = (128, 128)
PHOTO_PATH = "pepeketua_id"
LMDB_PATH = join(PHOTO_PATH, "lmdb")
ZIP_NAMES = [
    # "whareorino_a.zip",
    # "whareorino_b.zip",
    # "whareorino_c.zip",
    "whareorino_d.zip",
    # "pukeokahu.zip",
]
SQL_SERVER_STRING = "postgresql://lioruzan:nyudEce5@localhost/frogs"
WHAREORINO_EXCEL_FILE = join(
    PHOTO_PATH,
    "Whareorino frog monitoring data 2005 onwards CURRENT FILE - DOCDM-106978.xls",
)
PUKEOKAHU_EXCEL_FILE = join(
    PHOTO_PATH, "Pukeokahu Monitoring Data 2006 onwards - DOCDM-95563.xls"
)
LANDMARK_MODEL = "model_weights/landmark_model_714"
ROTATION_MODEL = "model_weights/rotation_model_weights_10"
IDENTIFY_MODEL = "model_weights/ep29_vloss0.0249931520129752_emb.ckpt"


def fetch_images_from_lmdb(keys: pd.Series) -> List[bytes]:
    with LmdbReader(LMDB_PATH) as reader:
        image_list = [
            reader.read_bytes(key) if key is not None else None for key in keys
        ]
        return image_list


def get_image_size(image_bytes: bytes) -> Tuple:
    with Image.open(BytesIO(image_bytes)) as im:
        return im.width, im.height


@functools.lru_cache
def get_dummy_image_bytes() -> bytes:
    memory_file = BytesIO()
    image = Image.new("RGB", (1, 1))
    image.save(memory_file, "JPEG")
    return memory_file.getvalue()


def prepare_batch(result_batch: Iterator[Sequence[Row]]) -> pd.DataFrame:
    batch_df = pd.DataFrame(result_batch)
    batch_df["image_bytes"] = fetch_images_from_lmdb(batch_df["lmdb_key"])
    # Insert dummy image instead of lines with no image
    batch_df["image_bytes"].where(
        batch_df["image_bytes"].notna(), get_dummy_image_bytes(), inplace=True
    )
    batch_df.loc[:, ["width_size", "height_size"]] = list(
        map(get_image_size, batch_df["image_bytes"])
    )
    return batch_df
