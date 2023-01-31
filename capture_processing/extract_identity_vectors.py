from typing import Dict

import faiss
import pandas as pd
import sqlalchemy as db
from loguru import logger
from tqdm import tqdm

from capture_processing.inference_model import get_inference_model
from inference_model import InferenceModel

# Suppress TensorFlow warnings
from utilities.utilities import (
    BATCH_SIZE,
    initialize_faiss_indices,
    prepare_batch,
    save_indices_to_lmdb,
    SQL_SERVER_STRING,
    SqlQuery,
    update_indices,
)


def fill_indices_with_identity_vectors_of_previous_captures(
    inference_model: InferenceModel,
    sql_server_string: str,
    indices: Dict[str, faiss.Index],
) -> Dict[str, faiss.Index]:
    with SqlQuery() as (connection, frogs):
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
            result_df = pd.DataFrame(result_batch)

            batch_df = prepare_batch(result_df)
            if batch_df is None:
                continue

            identity_vectors, ids, grids = inference_model.predict(batch_df)
            update_indices(indices, identity_vectors, ids, grids)
            break

        # Close db cursor after iteration ends
        cursor_result.close()

    return indices


def run():
    inference_model = get_inference_model()

    indices = initialize_faiss_indices()

    indices = fill_indices_with_identity_vectors_of_previous_captures(
        inference_model, SQL_SERVER_STRING, indices
    )

    for grid, index in indices.items():
        logger.info(f"Wrote {index.ntotal} vectors to {grid}'s Index")

    save_indices_to_lmdb(indices)
    logger.info("Done saving all identity vectors to disk.")


if __name__ == "__main__":
    run()
