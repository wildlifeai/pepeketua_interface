import sqlalchemy as db
from sqlalchemy import create_engine

from inference_model import InferenceModel
from utilities import utilities

# Suppress TensorFlow warnings
from utilities.utilities import (
    IDENTIFY_MODEL,
    LANDMARK_MODEL,
    prepare_batch,
    ROTATION_MODEL,
    SQL_SERVER_STRING,
)


def main(
    rotation_model: str,
    landmark_model: str,
    identify_model: str,
    sql_server_string: str,
):
    inference_model = InferenceModel(rotation_model, landmark_model, identify_model)

    engine = create_engine(sql_server_string)
    metadata = db.MetaData()
    with engine.connect() as connection:
        frogs = db.Table("frogs", metadata, autoload=True, autoload_with=connection)
        query = db.select([frogs])
        cursor_result = connection.execute(query)
        result_generator = cursor_result.partitions(size=utilities.BATCH_SIZE)

    for result_batch in result_generator:
        batch_df = prepare_batch(result_batch)
        results = inference_model.predict(batch_df)

    # Close db cursor after iteration ends
    cursor_result.close()


if __name__ == "__main__":
    main(ROTATION_MODEL, LANDMARK_MODEL, IDENTIFY_MODEL, SQL_SERVER_STRING)

    """
    import numpy as np
    import faiss
    
    # Create a random matrix of vectors to be indexed
    x = np.random.rand(100, 64).astype('float32')
    
    # Create an index
    index = faiss.IndexFlatL2(64) # 64 is the dimension of the vectors
    
    # Add the vectors to the index
    index.add(x)
    
    # Perform a search for nearest neighbors
    k = 10 # number of nearest neighbors to return
    query = np.random.rand(1, 64).astype('float32')
    D, I = index.search(query, k)
    
    # D is a matrix of distances
    # I is a matrix of indices of the nearest neighbors
    print(I)
    """
