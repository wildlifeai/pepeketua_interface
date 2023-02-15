from typing import Dict

import numpy as np


def get_bbox_coords(
    landmarks: Dict[str, float], delta: int = 5, width_height_format: bool = False
):
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
    xmin, ymin, xmax, ymax = xmin - delta, ymin - delta, xmax + delta, ymax + delta

    if width_height_format:
        # Convert bbox coords to (xmin, ymin, width, height) format, rounding them to closest int
        width = xmax - xmin
        height = ymax - ymin
        return tuple(map(round, (xmin, ymin, width, height)))
    else:
        return xmin, ymin, xmax, ymax
