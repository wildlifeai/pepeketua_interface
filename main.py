from loguru import logger

from inference_model import InferenceModel
from rotation_landmark_model.utilities import get_image_path, get_server_image_df


def main(rotation_model: str, landmark_model: str, identify_model: str, image_dir: str):
    inference_model = InferenceModel(rotation_model, landmark_model, identify_model)
    image_paths = get_image_path(image_dir)
    image_df = get_server_image_df(image_paths)
    results = inference_model.predict(image_df)
    logger.info(str(results))


if __name__ == "__main__":
    landmark_model = (
        "/Users/lioruzan/PycharmProjects/pepeketua_landmarks/model/landmark_model_714"
    )
    rotation_model = "/Users/lioruzan/PycharmProjects/pepeketua_landmarks/model/rotation_model_weights_10"
    identify_model = "/Users/lioruzan/PycharmProjects/pepeketua_identify/experiments/11062021_130149_cropped/ep29_vloss0.0249931520129752_emb.ckpt"
    image_dir = (
        "/Users/lioruzan/PycharmProjects/pepeketua_landmarks/images/frog_examples"
    )
    main(rotation_model, landmark_model, identify_model, image_dir)
