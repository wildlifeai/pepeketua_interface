from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from identify_model.utilities import preprocess_ims
from rotation_landmark_model import landmark_variables
from rotation_landmark_model.landmark_generator import LandMarkDataGenerator
from rotation_landmark_model.utilities import (
    debug_landmark_labels,
    fix_prediction_order,
    ORDERED_LANDMARK_UNNAMED_COLS,
    positive_deg_theta,
)
from utilities.utilities import (
    IDENTIFY_MODEL,
    LANDMARK_MODEL,
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

    def get_identity_vectors(self, image_df: pd.DataFrame) -> np.array:
        """
        Get identity vectors (embeddings) for a list of images with landmark predictions
        :param image_df:
        :return:
        """
        tf_images = preprocess_ims(image_df)
        identity_vectors = self.identity_model(tf_images)
        return identity_vectors.numpy()

    def predict(
        self, image_df: pd.DataFrame
    ) -> Tuple[np.array, Optional[np.array], pd.Series]:
        """
        Predict identity vectors from images with the following steps:
            1. Predict rotation of frog
            2. Rotate and predict landmarks for each image
            3. Use landmarks to predict identity vectors for each image
        :param image_df: df containing image data, will be processed in one batch.
        :return: Identity vectors for each image, Image ids (optional), Grid assignment for each image
        """

        """Start by predicting landmarks from images"""

        """This generator returns data of ALL of image_df in one batch. Batchifying needs to be done before this step"""
        generator = LandMarkDataGenerator(
            dataframe=image_df, x_col="image_bytes", batch_size=len(image_df)
        )
        rotation_model_images, regular_images = next(generator)

        rot_prediction = self.rotation_model(rotation_model_images).numpy()
        rot_prediction = fix_prediction_order(rot_prediction)
        pred_theta = np.arctan2(rot_prediction[:, 1], rot_prediction[:, 0])
        rot_prediction = positive_deg_theta(pred_theta)
        image_df["rotation"] = -rot_prediction

        # ############# LANDMARK MODEL ###############

        # The last column is the rotations for every image.
        # Rotate all images before feeding them into the landmark model
        rotations = image_df.loc[:, "rotation"].values[:, np.newaxis]
        rotated_regular_images = generator.rotate_images_specifically(
            rotations, regular_images
        )
        prediction = self.landmark_model(rotated_regular_images).numpy()

        prediction = fix_prediction_order(prediction)
        image_df.loc[:, ORDERED_LANDMARK_UNNAMED_COLS[:12]] = prediction

        # Resize landmark predictions to original image size
        pred_original_size = generator.create_final_labels(
            image_df[ORDERED_LANDMARK_UNNAMED_COLS]
        )

        # restore landmarks to original image size
        image_df[ORDERED_LANDMARK_UNNAMED_COLS[:-3]] = pred_original_size

        # save original images with keypoints on them for debugging purposes
        debug_images = False
        if debug_images:
            debug_landmark_labels(image_df, ORDERED_LANDMARK_UNNAMED_COLS[:-3])

        # Renaming columns to their full landmark names
        image_df.rename(
            columns=landmark_variables.change_column_name_dict, inplace=True
        )

        # Identity vector prediction
        identity_vectors = self.get_identity_vectors(image_df)

        # Return ids only if they exist in image_df (may be newly uploaded images with no ids)
        ids = image_df["id"].to_numpy() if "id" in image_df else None
        return identity_vectors, ids, image_df["Grid"]


@st.experimental_singleton
def get_inference_model():
    """
    Warning: keras models are not considered thread safe.
    If there will be many users this must be taken into consideration
    """
    return InferenceModel(ROTATION_MODEL, LANDMARK_MODEL, IDENTIFY_MODEL)
