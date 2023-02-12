from typing import Tuple

import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import tensorflow as tf
from imgaug import Keypoint

from rotation_landmark_model.server_image_generator import ServerFlowIterator
from rotation_landmark_model.utilities import ORDERED_LANDMARK_UNNAMED_COLS
from utilities.utilities import IMAGE_SIZE


class LandMarkDataGenerator(tf.keras.utils.Sequence):
    """Generator for augmenting images and keeping track of
    landmark points after augmentation
    """

    def __init__(
        self,
        dataframe,
        x_col,
        batch_size,
        normalize_y=False,
        preprocessing_function=None,
    ):
        """
        dataframe: dataframe holding the information

        x_col: column name of image paths

        rescale: float to multiply each element in the image, default 1/255

        normalize_y: Whether or not to divide point coordinates  by size of image
                     making point coordinates between 0-1, used when output layer's
                     activation is tanh etc.

        preprocessing_function: Preprocessing function to be past onto ImageDataGenerator

        Imgaug parameters for augmentation for more information
        visit imgaug documantations, all defualt values will
        lead to no augmentations, when using Tuples will chose a value
        in area uniformaly:

        scale: Scales image, default (1.0,1.0)
        translate_percent: Translates images by percentage, default (0,0)
        rotate: Rotates image, default (0,0)
        rotate_90: Rotates image by 90 degrees, default (0,0)
        shear: Shears image, default (0,0)
        multiply: The value with which to multiply the pixel values in each image, default (1, 1)
        multiply_per_channel: default 0 ,Whether to use (imagewise) the same sample(s) for all channels
                                (False) or to sample value(s) for each channel (True).
                                Setting this to True will therefore lead to different transformations per image and channel,
                                otherwise only per image. If this value is a float p, then for p percent of all images per_channel will be treated as True.
                                If it is a StochasticParameter it is expected to produce samples with values between 0.0 and 1.0,
                                where values >0.5 will lead to per-channel behaviour (i.e. same as True).

        """
        self.df = dataframe
        self.normalize_y = normalize_y
        self.preprocessing_function = preprocessing_function

        """Vectorizations used ONLY by create_final_labels() and ONLY in case of target size being IMAGE_SIZE"""
        # Vectorization function to create Keypoint objects for augmentation with resizing
        # image_size is (height, width)
        self._keypoint_vectorization_resize_back = np.vectorize(
            lambda x, y, im_height, im_width: Keypoint(x=x, y=y).project(
                IMAGE_SIZE, (im_height, im_width)
            )
        )
        self._keypoint_x_vectorization = np.vectorize(
            lambda x: (x.x / IMAGE_SIZE[0]) if self.normalize_y else x.x
        )
        self._keypoint_y_vectorization = np.vectorize(
            lambda y: (y.y / IMAGE_SIZE[1]) if self.normalize_y else y.y
        )

        # Image generator
        self.image_gen = ServerFlowIterator(
            df=self.df,
            x_col=x_col,
            batch_size=batch_size,
        )

    def __next__(self):
        return next(self.image_gen)

    def create_final_labels(self, labels: pd.DataFrame):
        # Resizes labels to original image size
        labels, image_size, rotations = self._fix_labels_get_image_size(labels)
        image_height, image_width = self._get_image_sizes_for_vectorizations(image_size)
        kps = self._keypoint_vectorization_resize_back(
            labels[:, :, 0], labels[:, :, 1], image_height, image_width
        )

        # imgaug Affine and Keypoint objects are used to rotate keypoints
        # This requires some fake images
        images = self._create_fake_images(image_height, image_width)
        images, kps = self.rotate_images_specifically(-rotations, images, kps=kps)

        points = np.array(kps)

        # Creating final x,y label pairs
        points_x = self._keypoint_x_vectorization(points)
        points_y = self._keypoint_y_vectorization(points)
        labels = np.reshape(
            np.dstack([points_x, points_y]), (labels.shape[0], labels.shape[1] * 2)
        )

        return labels

    def _create_fake_images(self, image_heights, image_widths):
        # For rotations images must be provided
        images = []

        for i in range(len(image_heights)):
            height = int(image_heights[i][0])
            width = int(image_widths[i][0])
            # Shape to feed to iaa.Affine
            images.append(np.ones((1, height, width, 1)))

        return images

    def _fix_labels_get_image_size(self, labels: pd.DataFrame) -> Tuple:
        rotations = labels["rotation"].to_numpy().reshape((len(labels), 1))
        labels = labels[ORDERED_LANDMARK_UNNAMED_COLS[:-1]]

        image_size = labels[["width_size", "height_size"]].to_numpy()
        labels = labels[ORDERED_LANDMARK_UNNAMED_COLS[:-3]].to_numpy()
        labels = np.reshape(labels, (labels.shape[0], int(labels.shape[1] / 2), 2))

        return labels, image_size, rotations

    def _get_image_sizes_for_vectorizations(
        self, image_size: np.array
    ) -> Tuple[np.array, np.array]:
        image_height = image_size[:, 1]
        image_height = np.reshape(image_height, (image_height.shape[0], 1))
        image_width = image_size[:, 0]
        image_width = np.reshape(image_width, (image_width.shape[0], 1))
        return image_height, image_width

    def rotate_images_specifically(self, rotations, images, kps=None):
        for i, rot in enumerate(rotations):
            # For when fixing predictions
            if type(images) == list:
                img = images[i]
            else:
                img = images[i : i + 1, :]

            if kps is not None:
                kps_img = [kps[i].tolist()]
                img, kps_img = iaa.Affine(rotate=rot)(images=img, keypoints=kps_img)
                kps[i] = np.array(kps_img[0])
            else:
                img = iaa.Affine(rotate=rot)(images=img)

            images[i] = img

        if kps is not None:
            return images, kps

        return images

    def __len__(self):
        # Always return all of the data in one batch
        return 1
