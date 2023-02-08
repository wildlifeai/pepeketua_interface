from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from capture_processing.inference_model import get_inference_model
from utilities.utilities import (
    get_image,
    get_row_and_image_by_id,
    get_usage_instructions,
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


def generate_images(df: pd.DataFrame, query_images: List[UploadedFile], k_nearest: int):

    image_names = tuple(image_file.name for image_file in query_images)
    image_indices = df["filepath"].str.endswith(image_names, na=False)

    # Get image contents and start process of identity vector collection
    batch_df = df[image_indices]
    batch_df = prepare_batch(batch_df)

    # Load inference model and calculate query id vectors
    inference_model = get_inference_model()
    new_id_vectors, _, grids = inference_model.predict(batch_df)

    grid_faiss_indices = load_faiss_indices_from_lmdb()

    # Display query-nn images by grid
    for grid_name, group in grids.groupby(grids):
        st.write(f"{grid_name} results.")

        query_indices = group.index
        query_vectors = new_id_vectors[query_indices]

        # Search for nearest neighbors to query vectors
        grid_faiss_index = grid_faiss_indices[grid_name]
        _, nn = grid_faiss_index.search(query_vectors, k=k_nearest)

        display_queries_and_nn_images(batch_df, nn, query_indices)


def display_queries_and_nn_images(
    batch_df: pd.DataFrame, nn: np.array, query_indices: pd.Index
):
    # For each query, display the query image on the left, then NN image slider on the right
    for j, batch_idx in enumerate(query_indices):
        # Ordering of containers and columns
        container = st.container()
        left_column, right_column = container.columns(2)

        # Show query image on left column
        im = get_image(batch_df.loc[batch_idx, "image_bytes"])
        left_column.image(im)

        # Display the NN image on top and the slider on the bottom
        top, bottom = right_column.empty(), right_column.empty()

        nn_indices = nn[j].tolist()
        nn_number = bottom.slider(
            label="Select nearest neighbor to view.",
            min_value=1,
            max_value=len(nn_indices),
            key=j,
        )

        # Render the selected nearest neighbor to the query
        _, image = get_row_and_image_by_id(nn_indices[nn_number - 1])
        top.image(image)


def write_imageless_rows(image_indices: pd.Series, full_uploaded_excel: pd.DataFrame):
    # Show rows without uploaded image match
    if len(full_uploaded_excel[image_indices]) != len(full_uploaded_excel):
        imageless_rows = full_uploaded_excel[~image_indices]
        st.write(
            f"{len(imageless_rows)} rows lack images: ",
            imageless_rows.head(),
        )


def main():

    # Show sidebar form
    with st.sidebar:
        with st.form(key="Uploader form"):
            excel_file = st.file_uploader(
                label="**Upload your excel sheet with new observations.**",
                label_visibility="visible",
            )

            query_image_files = st.file_uploader(
                label="**Upload frog pics**",
                label_visibility="visible",
                accept_multiple_files=True,
            )

            # Select K for K Nearest Neighbors
            k_nearest = st.slider(
                label="Select K value for K Nearest Neighbors",
                min_value=1,
                max_value=10,
                value=3,
            )

            submitted_knn = st.form_submit_button("Generate images")

    # Write out usage instructions at app start
    main_container = st.empty()
    main_container.write(get_usage_instructions())
    if st.session_state.get("session_started", False):
        main_container.empty()

    if submitted_knn or st.session_state.get("session_started", False):

        # Use this boolean to track if submitted_knn has been activated before
        # This allows image displays to persist after NN sliders have been changed
        if "session_started" not in st.session_state:
            st.session_state["session_started"] = True

        # For some reason ignoring the output of 'submitted_knn' works best
        try:
            df = pd.read_excel(excel_file, sheet_name=0)
        except Exception as e:
            st.write(
                f"Error loading {excel_file.name}. First sheet should be the one to use."
            )
            raise e

        # Collect and show results
        with main_container.container():
            match_images_to_rows(df, query_image_files)
            generate_images(df, query_image_files, k_nearest)


if __name__ == "__main__":
    main()
