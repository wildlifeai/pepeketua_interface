from io import BytesIO

import numpy as np
import pandas as pd
from keras_preprocessing.image import img_to_array
from PIL import Image

from utilities.utilities import (
    force_image_to_be_rgb,
    IMAGE_SIZE,
    ROT_IMAGE_SIZE,
    time_it,
)


class ServerFlowIterator:
    def __init__(
        self,
        df: pd.DataFrame,
        x_col: str,
        batch_size: int,
    ):
        self.df = df
        self.n = len(df)
        self.x_col = x_col
        self.batch_size = batch_size
        self.rescale = 1 / 255
        self.batch_index = 0
        self.total_batches_seen = 0
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)

    def __getitem__(self, idx: int) -> np.array:
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[
            self.batch_size * idx : self.batch_size * (idx + 1)
        ]
        return self._get_batches_of_transformed_samples(index_array)

    @time_it
    def _get_batches_of_transformed_samples(self, index_array: np.array):
        # generate batch array including color depth channel
        rotate_batch_x = np.zeros(
            (len(index_array),) + ROT_IMAGE_SIZE + (3,), dtype=None
        )
        regular_batch_x = np.zeros((len(index_array),) + IMAGE_SIZE + (3,), dtype=None)

        for i, j in enumerate(index_array):
            # load, resize and rescale image to target size and scale
            with Image.open(BytesIO(self.df.iloc[j][self.x_col])) as image:
                image = force_image_to_be_rgb(image)
                rotation_model_image = image.resize(
                    ROT_IMAGE_SIZE[0], Image.Resampling.NEAREST
                )
                regular_image = image.resize(IMAGE_SIZE[0], Image.Resampling.NEAREST)

            rotate_image_arr = np.array(
                [img_to_array(rotation_model_image)]
            )  # Convert single image to a batch.
            regular_image_arr = np.array(
                [img_to_array(regular_image)]
            )  # Convert single image to a batch.

            rotate_image_arr *= self.rescale  # rescale image by factor
            regular_image_arr *= self.rescale  # rescale image by factor

            rotate_batch_x[i] = rotate_image_arr
            regular_batch_x[i] = regular_image_arr

        return rotate_batch_x, regular_batch_x

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        self.reset()
        while 1:
            if self.batch_index == 0:
                self._set_index_array()

            if self.n == 0:
                current_index = 0
            else:
                current_index = (self.batch_index * self.batch_size) % self.n

            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0

            yield self.index_array[current_index : current_index + self.batch_size]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        return self

    def next(self):
        index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

    def __next__(self):
        return self.next()
