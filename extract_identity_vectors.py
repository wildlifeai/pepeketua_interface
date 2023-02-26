from typing import Dict

import faiss
import pandas as pd
import sqlalchemy as db
from loguru import logger
from tqdm import tqdm

from inference_model.inference_model import get_inference_model, InferenceModel
from utilities.utilities import (
    BATCH_SIZE,
    fetch_images_from_lmdb,
    initialize_faiss_indices,
    save_faiss_indices_to_lmdb,
    SqlQuery,
    update_indices,
)


def fill_indices_with_identity_vectors_of_previous_captures(
    inference_model: InferenceModel,
    indices: Dict[str, faiss.Index],
) -> Dict[str, faiss.Index]:
    with SqlQuery() as (connection, frogs):
        query = db.select(
            [frogs.columns.id, frogs.columns.lmdb_key, frogs.columns.Grid]
        )
        cursor_result = connection.execute(query)
        result_generator = cursor_result.partitions(size=BATCH_SIZE)

        # Manually update tqdm loop to be able to count skipped steps
        total_batches = (
            cursor_result.rowcount // BATCH_SIZE + 1
            if (cursor_result.rowcount % BATCH_SIZE) > 0
            else 0
        )
        with tqdm(total=total_batches) as pbar:
            for result_batch in result_generator:

                result_df = pd.DataFrame(result_batch)
                image_bytes = pd.Series(
                    fetch_images_from_lmdb(result_df["lmdb_key"]), index=result_df.index
                )
                batch_df, image_bytes = inference_model.prepare_batch(
                    result_df, image_bytes
                )
                if batch_df is None:
                    pbar.update(1)
                    continue

                identity_vectors, ids, grids = inference_model.predict(
                    batch_df, image_bytes
                )
                update_indices(indices, identity_vectors, ids, grids)
                pbar.update(1)

        # Close db cursor after iteration ends
        cursor_result.close()

    return indices


def run():
    inference_model = get_inference_model()

    indices = initialize_faiss_indices()

    indices = fill_indices_with_identity_vectors_of_previous_captures(
        inference_model, indices
    )

    for grid, index in indices.items():
        logger.info(f"Wrote {index.ntotal} vectors to {grid}'s Index")

    save_faiss_indices_to_lmdb(indices)
    logger.info("Done saving all identity vectors to disk.")


if __name__ == "__main__":
    run()
