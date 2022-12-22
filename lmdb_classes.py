import io
from typing import List

import lmdb
from loguru import logger
from PIL import Image


class ImageRecord(object):
    def _init_(self, buffer: io.BytesIO) -> None:
        """
        This class is used when reading or loading images from the disk.
        """
        assert buffer is not None, "Invalid buffer given!"
        self.image_buf = buffer

    def get_image(self):
        """
        :return: a PIL image
        """
        return Image.open(self.image_buf)

    def serialize(self):
        return self.image_buf.getvalue()

    @staticmethod
    def deserialize(record_bytestring: bytes) -> "ImageRecord":
        return ImageRecord(buffer=io.BytesIO(record_bytestring))


class LmdbReader(object):
    def _init_(self, data_path: str, num_workers: int = 8):
        self.env = lmdb.open(
            data_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_spare_txns=num_workers,
        )

    def read_image(self, record_index: int):
        return self.read_image_record(record_index).get_image()

    def read_image_record(self, record_index: int) -> ImageRecord:
        with self.env.begin(write=False) as txn:
            buffer = txn.get(f"{record_index:09}".encode())
            assert (
                buffer is not None
            ), f"The following record was not found in the LMDB: {record_index}"
            return ImageRecord.deserialize(buffer)


class LmdbWriter(object):
    def _init_(self, output_path: str, cache_size: int = 2000) -> None:

        self.env = None
        self.output_path = output_path
        self.cache_size = cache_size
        self.env = self.open_env()
        self.cache = {}

        with self.env.begin(write=False) as txn:
            self.append_mode = txn.stat()["entries"] > 0

        # set item count
        if self.append_mode is True:
            logger.info(f"LmdbWriter append mode ON. appending to {output_path}.")
            with self.env.begin(write=False) as txn:
                self.count = int(txn.get(b"num_samples"))
        else:
            self.count = 0

    def open_env(self):
        return lmdb.open(self.output_path, map_size=10**12)  # 1TB max map size

    def close_env(self):
        self.env.close()

    def add_data(self, records: List[ImageRecord]):
        old_count = self.count
        for record in records:
            key = f"{self.count:09}".encode()
            self.cache[key] = record.serialize()
            self.count += 1

            if self.count % self.cache_size == self.cache_size - 1:
                self.write_cache()

        # write remaining samples to lmdb
        self.write_cache()

        if self.count == old_count:
            logger.warning(
                "LmdbWriter didn't write any samples to disk! Something must be wrong."
            )
        return

    def write_cache(self) -> None:
        # record current number of samples before writing to lmdb
        self.cache[b"num_samples"] = str(self.count).encode()

        with self.env.begin(write=True) as txn:
            for k, v in self.cache.items():
                o = txn.put(k, v)
                if o is False:
                    logger.error("Error writing to lmdb or something.")
        self.clear_cache()
        return

    def clear_cache(self) -> None:
        self.cache.clear()
