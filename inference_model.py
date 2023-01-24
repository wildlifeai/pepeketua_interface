from io import BytesIO

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from PIL import Image

from identify_model.utilities import preprocess_im
from rotation_landmark_model import landmark_variables
from rotation_landmark_model.landmark_generator import LandMarkDataGenerator
from rotation_landmark_model.utilities import (
    debug_landmark_labels,
    fix_prediction_order,
    ORDERED_LANDMARK_UNNAMED_COLS,
    positive_deg_theta,
)
from utilities.utilities import (
    IMAGE_SIZE,
    ROT_IMAGE_SIZE,
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

    def get_embeddings(self, image_df: pd.DataFrame) -> np.array:
        """
        Get identity vectors (embeddings) for a list of images with landmark predictions
        :param image_df:
        :return:
        """
        embs = []
        for row in image_df.to_dict("records"):
            with Image.open(BytesIO(row["image_bytes"])) as im:
                im_tf = preprocess_im(im, landmarks=row)
            im_embs = self.identity_model(im_tf)
            embs.append(im_embs.numpy()[0])

        return embs

    def predict(self, image_df: pd.DataFrame) -> np.array:
        """
        Predict identity vectors from images with the following steps:
            1. Predict rotation of frog
            2. Rotate and predict landmarks for each image
            3. Use landmarks to predict identity vectors for each image
        :param image_df:
        :return: Identity vectors for each image
        """

        """Start by predicting landmarks from images"""
        batch_size = len(image_df)
        gpred_rot = LandMarkDataGenerator(
            dataframe=image_df,
            x_col="image_bytes",
            y_col=["width_size", "height_size"],
            color_mode="rgb",
            target_size=ROT_IMAGE_SIZE,
            batch_size=batch_size,
            training=False,
            resize_points=True,
            specific_rotations=False,
        )

        logger.info("Predicting rotation")
        rot_prediction = self.rotation_model.predict(gpred_rot)
        rot_prediction = fix_prediction_order(rot_prediction)
        pred_theta = np.arctan2(rot_prediction[:, 1], rot_prediction[:, 0])
        rot_prediction = positive_deg_theta(pred_theta)
        image_df["rotation"] = -rot_prediction

        gpred = LandMarkDataGenerator(
            dataframe=image_df,
            x_col="image_bytes",
            y_col=["width_size", "height_size", "rotation"],
            color_mode="rgb",
            target_size=IMAGE_SIZE,
            batch_size=batch_size,
            training=False,
            resize_points=True,
            specific_rotations=True,
        )

        logger.info("Predicting")
        prediction = self.landmark_model.predict(gpred)
        prediction = fix_prediction_order(prediction)
        pred_df = pd.concat(
            [
                image_df.image_bytes,
                pd.DataFrame(prediction),
                image_df.width_size,
                image_df.height_size,
                image_df.rotation,
            ],
            axis=1,
        )

        # resize predictions to original image size
        pred_original_size = gpred.create_final_labels(
            pred_df[ORDERED_LANDMARK_UNNAMED_COLS]
        )

        # restore landmarks to original image size
        pred_df[ORDERED_LANDMARK_UNNAMED_COLS[:-3]] = pred_original_size

        # save original images with keypoints on them for debugging purposes
        debug_images = False
        if debug_images:
            debug_landmark_labels(pred_df, ORDERED_LANDMARK_UNNAMED_COLS[:-3])

        # Renaming columns to their full landmark names
        pred_df.rename(columns=landmark_variables.change_column_name_dict, inplace=True)

        # Identity vector prediction
        identity_embeddings = self.get_embeddings(pred_df)
        pred_df.drop(columns="image_bytes", inplace=True)
        pred_df.loc[:, "embedding"] = identity_embeddings

        return pred_df
