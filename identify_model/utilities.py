import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


def get_bbox_corners(landmarks, delta=5):
    x_annots = [
        landmarks["x_Left_eye"],
        landmarks["x_Left_front_leg"],
        landmarks["x_Right_eye"],
        landmarks["x_Right_front_leg"],
        landmarks["x_Tip_of_snout"],
        landmarks["x_Vent"],
    ]
    y_annots = [
        landmarks["y_Left_eye"],
        landmarks["y_Left_front_leg"],
        landmarks["y_Right_eye"],
        landmarks["y_Right_front_leg"],
        landmarks["y_Tip_of_snout"],
        landmarks["y_Vent"],
    ]

    xmin, xmax, ymin, ymax = (
        np.min(x_annots),
        np.max(x_annots),
        np.min(y_annots),
        np.max(y_annots),
    )
    annots_extreme_4 = xmin - delta, ymin - delta, xmax + delta, ymax + delta
    return annots_extreme_4


def preprocess_im(im: Image, landmarks: pd.Series) -> np.array:
    crop_bbox = get_bbox_corners(landmarks, delta=5)
    im = im.crop(crop_bbox)
    im = np.array(im)
    im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
    im = tf.convert_to_tensor(im, dtype=tf.float32)
    im = tf.expand_dims(im, axis=0)
    return im
