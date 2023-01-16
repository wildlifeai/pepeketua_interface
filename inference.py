import numpy as np
import pandas as pd
import tensorflow as tf


class InferenceModel:
    def __init__(
        self,
        rotation_model_weights: str,
        landmark_model_weights: str,
        identify_model_weights: str,
    ):
        self.rotation_model = tf.keras.models.load_model(rotation_model_weights)
        self.landmark_model = tf.keras.models.load_model(landmark_model_weights)
        self.identify_model = tf.keras.models.load_model(identify_model_weights)

    def predict(self, frog_df: pd.DataFrame) -> np.array:
        pass
