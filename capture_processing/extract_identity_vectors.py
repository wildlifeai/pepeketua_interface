from typing import Dict

import faiss
import sqlalchemy as db
from loguru import logger
from sqlalchemy import create_engine
from tqdm import tqdm

from inference_model import InferenceModel

# Suppress TensorFlow warnings
from utilities.utilities import (
    BATCH_SIZE,
    IDENTIFY_MODEL,
    initialize_faiss_indices,
    LANDMARK_MODEL,
    prepare_batch,
    ROTATION_MODEL,
    save_indices_to_lmdb,
    SQL_SERVER_STRING,
    update_indices,
)


def fill_indices_with_identity_vectors_of_previous_captures(
    inference_model: InferenceModel,
    sql_server_string: str,
    indices: Dict[str, faiss.Index],
) -> Dict[str, faiss.Index]:
    engine = create_engine(sql_server_string)
    metadata = db.MetaData()

    with engine.connect() as connection:
        frogs = db.Table("frogs", metadata, autoload=True, autoload_with=connection)

        # create a select statement
        query = db.select(
            [frogs.columns.id, frogs.columns.lmdb_key, frogs.columns.Grid]
        )
        cursor_result = connection.execute(query)
        result_generator = cursor_result.partitions(size=BATCH_SIZE)

    total_batches = (
        cursor_result.rowcount // BATCH_SIZE + 1
        if (cursor_result.rowcount % BATCH_SIZE) > 0
        else 0
    )
    for result_batch in tqdm(result_generator, total=total_batches):
        batch_df = prepare_batch(result_batch)
        if batch_df is None:
            continue

        identity_vectors, ids, grids = inference_model.predict(batch_df)
        update_indices(indices, identity_vectors, ids, grids)

    # Close db cursor after iteration ends
    cursor_result.close()

    return indices


def run(
    rotation_model: str,
    landmark_model: str,
    identify_model: str,
    sql_server_string: str,
):
    inference_model = InferenceModel(rotation_model, landmark_model, identify_model)

    indices = initialize_faiss_indices()

    indices = fill_indices_with_identity_vectors_of_previous_captures(
        inference_model, sql_server_string, indices
    )

    for grid, index in indices.items():
        logger.info(f"Wrote {index.ntotal} vectors to {grid}'s Index")

    save_indices_to_lmdb(indices)
    logger.info("Done saving all identity vectors to disk.")


if __name__ == "__main__":
    run(ROTATION_MODEL, LANDMARK_MODEL, IDENTIFY_MODEL, SQL_SERVER_STRING)
