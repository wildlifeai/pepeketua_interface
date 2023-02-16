from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from inference_model.inference_model import get_inference_model
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
    query_df: pd.DataFrame,
    query_images: List[UploadedFile],
    k_nearest: int,
    scaling_factor: Tuple[int, int],
):
    query_df, query_image_bytes = match_images_to_rows(query_df, query_images)

    grids, new_id_vectors, query_image_bytes = calculate_query_id_vectors(
        query_df, query_image_bytes
    )

    nn_ids, distances_list, query_indices = [], [], []

    # Search for nearest neighbors to query vectors
    grid_faiss_indices = load_faiss_indices_from_lmdb()
    for grid_name, group in grids.groupby(grids):
        current_grid_query_indices = group.index
        query_vectors = new_id_vectors[current_grid_query_indices]

        grid_faiss_index = grid_faiss_indices[grid_name]
        distances, nn = grid_faiss_index.search(query_vectors, k=k_nearest)

        distances_list.append(distances)
        nn_ids.append(nn)
        query_indices.append(current_grid_query_indices)

    distances = np.concatenate(distances_list)
    nn_ids = np.concatenate(nn_ids)
    query_indices = np.concatenate(query_indices)
    nn_df = pd.DataFrame(
        data={"distances": distances.tolist(), "nn_ids": nn_ids.tolist()},
        index=query_indices,
    ).sort_index()

    display_queries_and_nn_images(query_df, query_image_bytes, nn_df, scaling_factor)


@st.experimental_memo
def calculate_query_id_vectors(df, query_image_bytes):
    # Load inference model and calculate query id vectors
    inference_model = get_inference_model()
    batch_df, query_image_bytes = inference_model.prepare_batch(df, query_image_bytes)
    new_id_vectors, _, grids = inference_model.predict(batch_df, query_image_bytes)
    return grids, new_id_vectors, query_image_bytes


def display_queries_and_nn_images(
    query_df: pd.DataFrame,
    query_image_bytes: pd.Series,
    nn_df: pd.DataFrame,
    scaling_factor: Tuple[int, int],
):
    # Set up main screen- two columns with three sections in each
    container = st.container()
    left_column, right_column = container.columns(2)
    top_right, mid_right, bottom_right = (
        right_column.empty(),
        right_column.empty(),
        right_column.container(),
    )
    top_left, mid_left, bottom_left = (
        left_column.empty(),
        left_column.empty(),
        left_column.container(),
    )

    # Left column showing the query images
    query_index = mid_left.selectbox(
        label="Select query to show",
        options=nn_df.index,
        format_func=lambda q: query_df.loc[q, "filepath"],
    )
    top_left.image(
        get_image(query_image_bytes.loc[query_index], scaling_factor=scaling_factor),
    )
    bottom_left_expander = bottom_left.expander("Show info")
    bottom_left_expander.dataframe(
        query_df.loc[query_index, :], use_container_width=True
    )

    # Right column showing nearest neighbor images
    nn_indices = nn_df.loc[query_index, "nn_ids"]
    nn_number = mid_right.select_slider(
        label="Select nearest neighbor to view.",
        options=range(1, len(nn_indices) + 1),
    )
    captured_frog_row, captured_frog_image_bytes = get_row_and_image_by_id(
        nn_indices[nn_number - 1]
    )
    top_right.image(get_image(captured_frog_image_bytes, scaling_factor=scaling_factor))
    bottom_right_expander = bottom_right.expander("Show info")
    bottom_right_expander.dataframe(captured_frog_row.T, use_container_width=True)


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
