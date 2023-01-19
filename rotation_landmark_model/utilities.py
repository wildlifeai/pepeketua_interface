import math
import os
from io import BytesIO
from os.path import join
from typing import List

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from rotation_landmark_model import landmark_variables


def get_image_path(dir_path: str) -> List[str]:
    """Make list of all images in dir_path directory"""
    image_paths = []
    for (dir_path, dir_names, file_names) in os.walk(dir_path):
        for file_name in file_names:
            file_path = os.sep.join([dir_path, file_name])
            if is_image(file_path):
                image_paths.append(file_path)

    return image_paths


def get_image_df(image_paths: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(image_paths, columns=["image_path"])

    # Using loop and not apply because this is considerably faster
    image_sizes = []
    for im_path in df.image_path:
        im = cv2.imread(im_path)
        im_size = im.shape[:2]
        image_sizes.append(im_size)

    image_sizes = np.array(image_sizes)
    df["width_size"] = image_sizes[:, 1]
    df["height_size"] = image_sizes[:, 0]

    return df


def get_server_image_df(image_paths: List[str]) -> pd.DataFrame:
    image_df = get_image_df(image_paths)
    image_bytes = []
    for im_path in image_df.image_path:
        with open(im_path, "rb") as file:
            image_bytes.append(file.read())
    image_df["image_path"] = image_bytes
    # restore order of labels, it is important for rest of code
    return image_df


def is_image(file_path):
    try:
        im = Image.open(file_path)
        im.close()
        return True
    except Exception as e:
        return False


def fix_prediction_order(prediction):
    """I have no idea why this is necessary"""
    if len(prediction) <= landmark_variables.BATCH_SIZE:
        return prediction
    else:
        return np.roll(prediction, landmark_variables.BATCH_SIZE, axis=0)


# Converts rad to theta in range 0 - 360
def positive_deg_theta(theta):
    theta = np.rad2deg(theta)
    return np.mod(theta + 360, 360)


def debug_landmark_labels(pred_df: pd.DataFrame, labels_column_names: List[str]):
    # Save predictions on original size
    for j, image_path in enumerate(pred_df.image_path):
        if isinstance(image_path, str):
            im = cv2.imread(image_path)
        else:
            with Image.open(BytesIO(image_path)) as image_file:
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
