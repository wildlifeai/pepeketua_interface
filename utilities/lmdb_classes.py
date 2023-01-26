import lmdb
from loguru import logger


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
        return self

    def read(self, key: bytes) -> bytes:
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

    def add(self, key: bytes, value: bytes) -> None:
        self.cache[key] = value
        self.count += 1
        if self.count % self.cache_size == self.cache_size - 1:
            self.write_cache()

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
