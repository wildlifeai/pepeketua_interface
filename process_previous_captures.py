import extract_identity_vectors
from previous_capture_processing import clean_save_old_capture_data

if __name__ == "__main__":
    """Run steps to index identity vectors of all previously captured frogs"""
    clean_save_old_capture_data.run()
    extract_identity_vectors.run()
