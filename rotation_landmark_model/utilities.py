import math
from io import BytesIO
from os.path import join
from typing import List

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from utilities.utilities import BATCH_SIZE

# Columns in prediction df, ordered
ORDERED_LANDMARK_UNNAMED_COLS = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    "width_size",
    "height_size",
    "rotation",
]


def is_image(file_path):
    try:
        im = Image.open(file_path)
        im.close()
        return True
    except Exception:
        return False


def fix_prediction_order(prediction):
    """I have no idea why this is necessary"""
    if len(prediction) <= BATCH_SIZE:
        return prediction
    else:
        return np.roll(prediction, BATCH_SIZE, axis=0)


# Converts rad to theta in range 0 - 360
def positive_deg_theta(theta):
    theta = np.rad2deg(theta)
    return np.mod(theta + 360, 360)


def debug_landmark_labels(pred_df: pd.DataFrame, labels_column_names: List[str]):
    # Save predictions on original size
    for j, image_bytes in enumerate(pred_df["image_bytes"]):
        with Image.open(BytesIO(image_bytes)) as image_file:
            im = cv2.cvtColor(np.array(image_file), cv2.COLOR_RGB2BGR)
        labels = pred_df.iloc[j][labels_column_names].to_list()
        thickness = math.ceil(pred_df.iloc[j].height_size / 224) + 1
        _ = show_labels(im, labels, radius=1, thickness=thickness)


def show_labels(
    img,
    labels,
    labels_real=None,
    radius=5,
    thickness=1,
    radius_real=10,
    color=(0, 0, 255),
    color_real=(0, 255, 0),
):
    for i in range(0, len(labels), 2):
        point = np.round([labels[i], labels[i + 1]]).astype(int)
        point = tuple(point)
        img = cv2.circle(img, point, radius, color, thickness)
        if labels_real is not None:
            point = np.round([labels_real[i], labels_real[i + 1]]).astype(int)
            point = tuple(point)
            img = cv2.circle(img, point, radius, color_real, thickness)
            img = cv2.circle(img, point, radius_real, color_real, thickness)

    # Save the image
    nn = np.random.randint(0, 1000)
    image_path = join("landmark_images", "new_image_{0}.jpg").format(nn)
    cv2.imwrite(image_path, img)
    # show_image(img)
    return img


def show_image(img):
    cv2.imshow("", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
