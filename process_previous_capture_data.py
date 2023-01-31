from loguru import logger

from capture_processing import clean_save_old_capture_data, extract_identity_vectors

if __name__ == "__main__":
    # Log to disk
    logger.add("parse_previous_capture_data.log")

    """Run steps to index identity vectors of all previously captured frogs"""
    clean_save_old_capture_data.run()
    extract_identity_vectors.run()
