from io import BytesIO
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from keras_preprocessing.image import img_to_array
from PIL import Image

from pipelines.identify_model.utilities import get_bbox_coords
from pipelines.rotation_landmark_model.rotation_landmark_manipulator import (
    create_final_labels,
    rotate_images_specifically,
)
from pipelines.rotation_landmark_model.utilities import (
    debug_landmark_labels,
    fix_prediction_order,
    LANDMARK_COL_NAMES,
    positive_deg_theta,
)
from utilities.utilities import (
    force_image_to_rgb,
    IDENTIFY_MODEL,
    jpeg,
    KP_ID_MODEL_INPUT_IMAGE_SIZE,
    LANDMARK_MODEL,
    ROT_MODEL_INPUT_IMAGE_SIZE,
    ROTATION_MODEL,
)


class InferenceModel:
    def __init__(
        self,
        rotation_model_weights: str,
        landmark_model_weights: str,
        identify_model_weights: str,
    ):
        self.rotation_model = tf.keras.models.load_model(rotation_model_weights)
        self.landmark_model = tf.keras.models.load_model(landmark_model_weights)
        self.identity_model = tf.keras.models.load_model(identify_model_weights)
        self.rescale = 1 / 255

    def prepare_batch(
        self, batch_df: pd.DataFrame, image_bytes: pd.Series
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Get and return data and images for all rows that have non-NaN "filepath" entries, None if there aren't any.
        :param batch_df: df containing all frog information, whether new or from previous captures.
        :param image_bytes: series containing all images in batch.

        :return:
        """
        # Skip batch if there aren't any pictures in it
        if bool(image_bytes.any()) is False:
            return None, None

        # Copy to get rid of that SettingWithCopyWarning
        batch_df = batch_df.copy()

        # Filter rows with no images
        batch_df = batch_df[image_bytes.notna()]
        image_bytes = image_bytes[image_bytes.notna()]

        # Populate size columns
        width_height_list = list(map(self.get_image_size, image_bytes))
        batch_df.loc[:, ["width_size", "height_size"]] = pd.DataFrame(
            data=width_height_list,
            columns=["width_size", "height_size"],
            index=batch_df.index,
        )

        # For later calculations
        batch_df.reset_index(drop=True, inplace=True)
        return batch_df, image_bytes

    @staticmethod
    def get_image_size(image_bytes: bytes) -> Tuple:
        try:
            # Decode the JPEG header to get the image dimensions
            width, height, _, _ = jpeg.decode_header(image_bytes)
        except Exception:
            # In case file is not JPEG but some other format
            with Image.open(BytesIO(image_bytes)) as image_file:
                width, height = image_file.size
        return width, height

    @staticmethod
    def _transform_single_image(single_image_bytes: bytes, rescale: float = 1 / 255):
        """Load, resize and rescale image to target size and scale of kp/id/rotation models"""
        full_size_image = force_image_to_rgb(Image.open(BytesIO(single_image_bytes)))

        # Resize full size image to the sizes the models accept
        rotation_model_image = full_size_image.resize(
            ROT_MODEL_INPUT_IMAGE_SIZE, Image.Resampling.NEAREST
        )
        kp_id_model_image = full_size_image.resize(
            KP_ID_MODEL_INPUT_IMAGE_SIZE, Image.Resampling.NEAREST
        )

        # Convert single image to batch format.
        rotate_image_arr = img_to_array(rotation_model_image)[np.newaxis, :]
        kp_id_model_image_arr = img_to_array(kp_id_model_image)[np.newaxis, :]

        # rescale images by rescale factor
        rotate_image_arr *= rescale
        kp_id_model_image_arr *= rescale

        return full_size_image, rotate_image_arr, kp_id_model_image_arr

    def get_batches_of_transformed_images_for_models(
        self, image_bytes: pd.Series
    ) -> Tuple[np.array, np.array, np.array]:
        images_bytes = image_bytes.to_list()

        full_size_batch, rotate_batch, kp_id_batch = zip(
            *list(map(self._transform_single_image, images_bytes))
        )

        rotate_batch = np.concatenate(rotate_batch)
        kp_id_batch = np.concatenate(kp_id_batch)

        return rotate_batch, kp_id_batch, full_size_batch

    @staticmethod
    def preprocess_single_image_for_id_model(full_size_image, landmarks):
        bbox_coords = get_bbox_coords(landmarks=landmarks)
        image = full_size_image.crop(bbox_coords)
        image = force_image_to_rgb(image)
        image = cv2.resize(
            np.array(image), KP_ID_MODEL_INPUT_IMAGE_SIZE, interpolation=cv2.INTER_AREA
        )
        tf_im = tf.convert_to_tensor(image, dtype=tf.float32)
        tf_im = tf.expand_dims(tf_im, axis=0)
        return tf_im

    def preprocess_ims_for_id_model(
        self, df: pd.DataFrame, full_size_images: List[Image.Image]
    ) -> tf.Tensor:

        df_records = df.to_dict("records")
        images_tf = list(
            map(
                self.preprocess_single_image_for_id_model,
                full_size_images,
                df_records,
            )
        )

        return tf.concat(images_tf, axis=0)

    def predict(
        self, df: pd.DataFrame, image_bytes: pd.Series
    ) -> Tuple[np.array, Optional[np.array], pd.Series]:
        """
        Predict identity vectors from images with the following steps:
            1. Predict rotations of frog images
            2. Rotate images and predict landmarks for them
            3. Use landmarks to predict identity vectors for each image

        :param df: df containing image data, will be processed in one batch.
        :param image_bytes: Series containing images to predict from in bytes
        :return: Identity vectors for each image, Image ids (optional), Grid assignment for each image
        """
        # Copy to avoid SettingWithCopyWarning warnings
        df = df.copy()
        (
            rotation_model_images,
            kp_model_images,
            full_size_images,
        ) = self.get_batches_of_transformed_images_for_models(image_bytes)

        # ############## ROTATION MODEL ##############

        rot_prediction = self.rotation_model(rotation_model_images).numpy()
        rot_prediction = fix_prediction_order(rot_prediction)
        pred_theta = np.arctan2(rot_prediction[:, 1], rot_prediction[:, 0])
        rot_prediction = positive_deg_theta(pred_theta)
        df.loc[:, "rotation"] = -rot_prediction

        # ############# LANDMARK MODEL ###############

        # Correct rotation of all images before feeding them into the landmark model
        rotations = df.loc[:, "rotation"].values[:, np.newaxis]
        rotated_regular_images = rotate_images_specifically(rotations, kp_model_images)
        prediction = self.landmark_model(rotated_regular_images).numpy()
        prediction = fix_prediction_order(prediction)
        df.loc[:, LANDMARK_COL_NAMES] = pd.DataFrame(
            data=prediction, columns=LANDMARK_COL_NAMES, index=df.index
        )

        # Resize landmark predictions from KP_ID_MODEL_INPUT_IMAGE_SIZE to original full image size
        pred_original_size = create_final_labels(df)

        # Restore landmarks to original image size
        df.loc[:, LANDMARK_COL_NAMES] = pd.DataFrame(
            data=pred_original_size, columns=LANDMARK_COL_NAMES, index=df.index
        )

        # save original images with keypoints on them for debugging purposes
        debug_images = False
        if debug_images:
            debug_landmark_labels(df, LANDMARK_COL_NAMES)

        # ################ IDENTITY MODEL ###################

        # Get identity vectors (embeddings) for a list of images with landmark predictions
        id_model_images = self.preprocess_ims_for_id_model(df, full_size_images)
        identity_vectors = self.identity_model(id_model_images).numpy()

        # Return ids only if they exist in image_df (may be newly uploaded images with no ids)
        ids = df["id"].to_numpy() if "id" in df else None
        return identity_vectors, ids, df["Grid"]


@st.experimental_singleton
def get_inference_model():
    """
    Warning: keras models are not considered thread safe.
    If there will be many users this must be taken into consideration
    """
    return InferenceModel(ROTATION_MODEL, LANDMARK_MODEL, IDENTIFY_MODEL)
