from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from capture_processing.inference_model import get_inference_model
from utilities.utilities import (
    DEFAULT_K_NEAREST,
    get_image,
    get_row_and_image_by_id,
    get_usage_instructions,
    jpeg,
    load_faiss_indices_from_lmdb,
)


def match_images_to_rows(
    df: pd.DataFrame, image_files: List[UploadedFile]
) -> Tuple[pd.DataFrame, pd.Series]:
    assert (
        "filepath" in df
    ), "Excel sheet must contain the column 'filepath', please re-upload the correct sheet"
    image_bytes = pd.Series(dtype=bytes)

    for image_file in image_files:
        image_index = df["filepath"].str.endswith(image_file.name, na=False)

        # Can replace this with usage of "do_frog_ids_match" column
        # (but then need to collect results differently in the end)
        assert sum(image_index) == 1, (
            f"'{image_file.name}' must only be referenced by ONE row, "
            f"currently referenced by the rows {df[image_index].index}: {df[image_index]}"
        )

        image_bytes.loc[image_index.idxmax()] = image_file.getvalue()

    # Return relevant rows in df
    df = df.loc[image_bytes.index]
    df, image_bytes = df.reset_index(drop=True), image_bytes.reset_index(drop=True)

    return df, image_bytes


def generate_images(
    df: pd.DataFrame,
    query_images: List[UploadedFile],
    k_nearest: int,
    scaling_factor: Tuple[int, int],
):
    df, image_bytes = match_images_to_rows(df, query_images)
    # Get image contents and start process of identity vector collection
    # Load inference model and calculate query id vectors
    inference_model = get_inference_model()
    batch_df, image_bytes = inference_model.prepare_batch(df, image_bytes)
    new_id_vectors, _, grids = inference_model.predict(batch_df, image_bytes)

    grid_faiss_indices = load_faiss_indices_from_lmdb()

    # Display query-nn images by grid
    for grid_name, group in grids.groupby(grids):
        st.write(f"{grid_name} results.")

        query_indices = group.index
        query_vectors = new_id_vectors[query_indices]

        # Search for nearest neighbors to query vectors
        grid_faiss_index = grid_faiss_indices[grid_name]
        _, nn = grid_faiss_index.search(query_vectors, k=k_nearest)

        display_queries_and_nn_images(image_bytes, nn, query_indices, scaling_factor)


def display_queries_and_nn_images(
    image_bytes: pd.Series,
    nn: np.array,
    query_indices: pd.Index,
    scaling_factor: Tuple[int, int],
):
    # For each query, display the query image on the left, then NN image slider on the right
    for j, batch_idx in enumerate(query_indices):

        # Ordering of containers and columns
        container = st.container()
        left_column, right_column = container.columns(2)

        # Show query image on left column
        im = get_image(image_bytes.loc[batch_idx], scaling_factor=scaling_factor)
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
        _, single_image_bytes = get_row_and_image_by_id(nn_indices[nn_number - 1])
        top.image(get_image(single_image_bytes, scaling_factor=scaling_factor))


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
                value=DEFAULT_K_NEAREST,
            )
            # Scale factor for loading images
            scaling_options = sorted(jpeg.scaling_factors)
            scaling_factor = st.select_slider(
                label="Select scaling factor for image display",
                options=scaling_options,
                format_func=lambda scale: f"{scale[0]}/{scale[1]}",
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
            generate_images(df, query_image_files, k_nearest, scaling_factor)


if __name__ == "__main__":
    main()
