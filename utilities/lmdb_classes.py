import io
from typing import List

import lmdb
from loguru import logger
from PIL import Image


class ImageRecord(object):
    def __init__(self, buffer: io.BytesIO) -> None:
        """
        This class is used when reading or loading images from the disk.
        """
        assert buffer is not None, "Invalid buffer given!"
        self.image_buf = buffer

    def get_image(self) -> Image:
        """
        :return: a PIL image
        """
        return Image.open(self.image_buf)

    def serialize(self) -> bytes:
        return self.image_buf.getvalue()

    @staticmethod
    def deserialize(record_bytestring: bytes) -> "ImageRecord":
        return ImageRecord(buffer=io.BytesIO(record_bytestring))


class LmdbReader(object):
    def __init__(self, data_path: str, num_workers: int = 8):
        self.data_path = data_path
        self.num_workers = num_workers

    def __enter__(self):
        self.environment = lmdb.open(
            self.data_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_spare_txns=self.num_workers,
        )

    def read_image(self, key: bytes) -> Image:
        return self.read_image_record(key).get_image()

    def read_image_record(self, key: bytes) -> ImageRecord:
        buffer = self.read_bytes(key)
        return ImageRecord.deserialize(buffer)

    def read_bytes(self, key: bytes) -> bytes:
        with self.environment.begin(write=False) as txn:
            buffer = txn.get(key)
            assert (
                buffer is not None
            ), f"The following record was not found in the LMDB: {key.decode()}"
            return buffer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.environment.close()


class LmdbWriter(object):
    def __init__(self, output_path: str, cache_size: int = 1000) -> None:
        self.environment = None
        self.count = 0
        self.output_path = output_path
        self.cache_size = cache_size
        self.cache = {}

    def open_environment(self):
        return lmdb.open(self.output_path, map_size=10**12)  # 1TB max map size

    def __enter__(self):
        """Start of context manager- open environment, set append mode"""
        self.environment = self.open_environment()
        return self

    def add_record(self, key: bytes, record: ImageRecord) -> None:
        self.cache[key] = record.serialize()
        self.count += 1
        if self.count % self.cache_size == self.cache_size - 1:
            self.write_cache()

    def add_records(self, keys: List[bytes], records: List[ImageRecord]) -> None:
        for key, record in zip(keys, records):
            self.add_record(key, record)

    def write_cache(self) -> None:
        with self.environment.begin(write=True) as txn:
            for k, v in self.cache.items():
                o = txn.put(k, v)
                if o is False:
                    logger.error("Error writing to lmdb or something.")
        self.cache.clear()
        return

    def close(self):
        """Close the writer- write remaining cache to disk and close the Environment"""
        self.write_cache()
        self.environment.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End of context, write final sample and close enviroments"""
        self.close()
