from loguru import logger

from capture_processing import clean_save_old_capture_data, extract_identity_vectors
from utilities.utilities import (
    IDENTIFY_MODEL,
    LANDMARK_MODEL,
    PHOTO_PATH,
    PUKEOKAHU_EXCEL_FILE,
    ROTATION_MODEL,
    SQL_SERVER_STRING,
    WHAREORINO_EXCEL_FILE,
    ZIP_NAMES,
)

if __name__ == "__main__":
    # Log to disk
    logger.add("parse_previous_capture_data.log")

    """Run steps to index identity vectors of all previously captured frogs"""
    clean_save_old_capture_data.run(
        PHOTO_PATH,
        ZIP_NAMES,
        WHAREORINO_EXCEL_FILE,
        PUKEOKAHU_EXCEL_FILE,
        SQL_SERVER_STRING,
    )
    extract_identity_vectors.run(
        ROTATION_MODEL, LANDMARK_MODEL, IDENTIFY_MODEL, SQL_SERVER_STRING
    )
