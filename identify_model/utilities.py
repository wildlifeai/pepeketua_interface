from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from utilities.utilities import force_image_to_be_rgb, IMAGE_SIZE


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


def preprocess_ims(image_df: pd.DataFrame) -> tf.Tensor:
    images_tf = []
    dict_df = image_df.to_dict("records")
    for row in dict_df:
        with Image.open(BytesIO(row["image_bytes"])) as im:
            im = force_image_to_be_rgb(im)
            crop_bbox = get_bbox_corners(row, delta=5)
            im = im.crop(crop_bbox)
            im = cv2.resize(np.array(im), IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            tf_im = tf.convert_to_tensor(im, dtype=tf.float32)
            tf_im = tf.expand_dims(tf_im, axis=0)
            images_tf.append(tf_im)
    return tf.concat(images_tf, axis=0)
