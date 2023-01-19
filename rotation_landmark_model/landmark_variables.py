"""Various global variables used in the inference process"""
BATCH_SIZE = 3
IMAGE_SIZE = (224, 224)
ROT_IMAGE_SIZE = (128, 128)

"""Landmark column names"""
x_left_eye = "x_Left_eye"
y_left_eye = "y_Left_eye"
x_left_front_leg = "x_Left_front_leg"
y_left_front_leg = "y_Left_front_leg"
x_right_eye = "x_Right_eye"
y_right_eye = "y_Right_eye"
x_right_front_leg = "x_Right_front_leg"
y_right_front_leg = "y_Right_front_leg"
x_tip_of_snout = "x_Tip_of_snout"
y_tip_of_snout = "y_Tip_of_snout"
x_vent = "x_Vent"
y_vent = "y_Vent"
image_path = "image_path"

"""Array used to infer order of landmark columns"""
COLS_DF_NAMES = [
    x_left_eye,
    y_left_eye,
    x_left_front_leg,
    y_left_front_leg,
    x_right_eye,
    y_right_eye,
    x_right_front_leg,
    y_right_front_leg,
    x_tip_of_snout,
    y_tip_of_snout,
    x_vent,
    y_vent,
]

"""Indices of landmark columns"""
x_left_eye_index = COLS_DF_NAMES.index(x_left_eye)
y_left_eye_index = COLS_DF_NAMES.index(y_left_eye)
x_left_front_leg_index = COLS_DF_NAMES.index(x_left_front_leg)
y_left_front_leg_index = COLS_DF_NAMES.index(y_left_front_leg)
x_right_eye_index = COLS_DF_NAMES.index(x_right_eye)
y_right_eye_index = COLS_DF_NAMES.index(y_right_eye)
x_right_front_leg_index = COLS_DF_NAMES.index(x_right_front_leg)
y_right_front_leg_index = COLS_DF_NAMES.index(y_right_front_leg)
x_tip_of_snout_index = COLS_DF_NAMES.index(x_tip_of_snout)
y_tip_of_snout_index = COLS_DF_NAMES.index(y_tip_of_snout)
x_vent_index = COLS_DF_NAMES.index(x_vent)
y_vent_index = COLS_DF_NAMES.index(y_vent)

"""dict used to convert index to landmark column name"""
change_column_name_dict = {i: COLS_DF_NAMES[i] for i in range(0, len(COLS_DF_NAMES))}
