from io import BytesIO
from typing import List

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile

from capture_processing.inference_model import get_inference_model
from utilities.utilities import (
    get_rows_and_images_from_ids,
    load_faiss_indices_from_lmdb,
    prepare_batch,
)


def match_images_to_rows(df: pd.DataFrame, image_files: List[UploadedFile]):
    assert (
        "filepath" in df
    ), "Excel sheet must contain the column 'filepath', please re-upload the correct sheet"
    for image_file in image_files:
        image_index = df["filepath"].str.endswith(image_file.name, na=False)

        assert sum(image_index) > 0, (
            f"'{image_file.name}' cannot be found in any 'filepath'! "
            f"Upload only images that have a valid 'filepath' reference"
        )

        # Can replace this with usage of "do_frog_ids_match" column
        # (but then need to collect results differently in the end)
        assert sum(image_index) <= 1, (
            f"'{image_file.name}' must only be referenced by ONE row, "
            f"currently referenced by the rows {df[image_index].index}: {df[image_index]}"
        )

        # Save image to the df
        df.loc[image_index, "image_bytes"] = image_file.getvalue()


def generate_images(df: pd.DataFrame):

    # Get image contents and start process of identity vector collection
    batch_df = df[image_indices]
    batch_df = prepare_batch(batch_df)

    # Load inference model and calculate query id vectors
    inference_model = get_inference_model()
    new_id_vectors, _, grids = inference_model.predict(batch_df)

    grid_faiss_indices = load_faiss_indices_from_lmdb()

    for grid_name, group in grids.groupby(grids):
        st.write(f"{grid_name} results.")

        query_indices = group.index
        query_vectors = new_id_vectors[query_indices]

        faiss_index = grid_faiss_indices[grid_name]
        _, nn = faiss_index.search(query_vectors, k=k_nearest)

        for j, batch_idx in enumerate(query_indices):

            container = st.container()
            col1, col2 = container.columns(2)

            empty1, empty2 = col2.empty(), col2.empty()

            with Image.open(BytesIO(batch_df.loc[batch_idx, "image_bytes"])) as image:
                col1.image(image)

            nn_indices = nn[j].tolist()
            nn_number = empty2.slider(
                label="Select nearest neighbor to view.",
                min_value=1,
                max_value=len(nn_indices),
                key=j,
            )

            _, images = get_rows_and_images_from_ids(nn_indices)
            empty1.image(images[nn_number - 1])


if __name__ == "__main__":
    excel_file = st.sidebar.file_uploader(
        label="**Upload your excel sheet with new observations.**",
        label_visibility="visible",
    )
    if excel_file:
        try:
            df = pd.read_excel(excel_file, sheet_name=0)
        except Exception as e:
            st.sidebar.write(
                f"Error loading {excel_file.name}. First sheet should be the one to use."
            )
            raise e

        image_files = st.sidebar.file_uploader(
            label="**Upload frog pics**",
            label_visibility="visible",
            accept_multiple_files=True,
        )

        if image_files:

            match_images_to_rows(df, image_files)

            # Collect and show results
            image_names = tuple(image_file.name for image_file in image_files)
            image_indices = df["filepath"].str.endswith(image_names, na=False)

            # Show rows without uploaded image match
            # if len(df[image_indices]) != len(df):
            #     imageless_rows = df[~image_indices]
            #     st.write(
            #         f"{len(imageless_rows)} rows lack images: ",
            #         imageless_rows.head(),
            #     )

            # Select K for K Nearest Neighbors
            k_nearest = st.sidebar.slider(
                label="Select K value for K Nearest Neighbors",
                min_value=1,
                max_value=10,
                value=3,
            )

            if st.sidebar.checkbox(label="Generate images"):
                generate_images(df)
