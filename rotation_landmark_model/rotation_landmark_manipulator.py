from typing import List, Optional, Tuple, Union

import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
from imgaug import Keypoint

from rotation_landmark_model.utilities import LANDMARK_COL_NAMES
from utilities.utilities import KP_ID_MODEL_INPUT_IMAGE_SIZE

_keypoint_vectorization_resize_back = np.vectorize(
    lambda x, y, im_height, im_width: Keypoint(x=x, y=y).project(
        KP_ID_MODEL_INPUT_IMAGE_SIZE, (im_height, im_width)
    )
)
_keypoint_x_vectorization = np.vectorize(lambda x: x.x)
_keypoint_y_vectorization = np.vectorize(lambda y: y.y)


def create_final_labels(df: pd.DataFrame):
    # Resizes labels to original image size
    keypoints, image_size, rotations = _parse_labels(df)
    image_height, image_width = _get_image_sizes_for_vectorizations(image_size)
    resized_keypoints = _keypoint_vectorization_resize_back(
        keypoints[:, :, 0], keypoints[:, :, 1], image_height, image_width
    )

    # imgaug Affine and Keypoint objects are used to rotate keypoints back to original positions
    # This requires some fake images in same size as original pictures
    fake_images = _create_fake_images(image_height, image_width)
    _, resized_rotated_keypoints = rotate_images_specifically(
        -rotations, fake_images, kps_list=resized_keypoints
    )

    resized_rotated_keypoints = np.array(resized_rotated_keypoints)

    # Creating final keypoints
    points_x = _keypoint_x_vectorization(resized_rotated_keypoints)
    points_y = _keypoint_y_vectorization(resized_rotated_keypoints)
    resized_rotated_keypoints = np.reshape(
        np.dstack([points_x, points_y]),
        (keypoints.shape[0], keypoints.shape[1] * 2),
    )

    return resized_rotated_keypoints


def _create_fake_images(image_heights, image_widths):
    # For rotations images must be provided
    images = []

    for i in range(len(image_heights)):
        height = int(image_heights[i][0])
        width = int(image_widths[i][0])
        # Shape to feed to iaa.Affine
        images.append(np.ones((1, height, width, 1)))

    return images


def _parse_labels(labels: pd.DataFrame) -> Tuple[np.array, np.array, np.array]:
    rotations = labels["rotation"].to_numpy().reshape((len(labels), 1))

    image_size = labels[["width_size", "height_size"]].to_numpy()

    keypoints_arr = labels[LANDMARK_COL_NAMES].to_numpy()
    keypoints_arr = np.reshape(
        keypoints_arr, (keypoints_arr.shape[0], int(keypoints_arr.shape[1] / 2), 2)
    )

    return keypoints_arr, image_size, rotations


def _get_image_sizes_for_vectorizations(
    image_size: np.array,
) -> Tuple[np.array, np.array]:
    image_height = image_size[:, 1]
    image_height = np.reshape(image_height, (image_height.shape[0], 1))
    image_width = image_size[:, 0]
    image_width = np.reshape(image_width, (image_width.shape[0], 1))
    return image_height, image_width


def _rotate_single_image(
    image: np.array, rotation: float, kps: Optional[List[Keypoint]]
):
    if kps is not None:
        rotated_image, rotated_kps = iaa.Affine(rotate=rotation)(
            images=image, keypoints=[kps]
        )
        rotated_kps = np.array(rotated_kps[0])
    else:
        rotated_image = iaa.Affine(rotate=rotation)(images=image)
        rotated_kps = None
    return rotated_image, rotated_kps


def rotate_images_specifically(
    rotations: np.array,
    images: Union[np.array, List[np.array]],
    kps_list: Optional[np.array] = None,
) -> Union[List[np.array], np.array]:

    rotations = rotations.tolist()

    if isinstance(images, list):
        # In this case the images list is of full sized images and have different shapes
        # So no concatenation possible or necessary
        concatenate_images = False
    else:
        images = [images[i : i + 1, :] for i in range(images.shape[0])]
        concatenate_images = True

    if kps_list is None:
        kps_list = [None] * len(rotations)
    else:
        kps_list = kps_list.tolist()

    rotated_images, rotated_kps = zip(
        *list(map(_rotate_single_image, images, rotations, kps_list))
    )

    if concatenate_images:
        rotated_images = np.concatenate(rotated_images)

    if rotated_kps[0] is not None:
        return rotated_images, rotated_kps
    else:
        return rotated_images


def rotate_images_specifically_deprecated(
    rotations: np.array,
    images: Union[np.array, List[np.array]],
    kps_list: Optional[List[Keypoint]] = None,
) -> Union[List[np.array], np.array]:
    for i, rot in enumerate(rotations):

        if isinstance(images, list):
            img = images[i]
        else:
            img = images[i : i + 1, :]

        if kps_list is not None:
            kps_img = [kps_list[i].tolist()]
            img, kps_img = iaa.Affine(rotate=rot)(images=img, keypoints=kps_img)
            kps_list[i] = np.array(kps_img[0])
        else:
            img = iaa.Affine(rotate=rot)(images=img)

        images[i] = img

    if kps_list is not None:
        return images, kps_list

    return images
