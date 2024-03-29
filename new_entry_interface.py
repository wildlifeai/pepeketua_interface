from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from inference_model.inference_model import get_inference_model
from utilities.utilities import (
    compare_bincodes,
    DEFAULT_KNN_VALUE,
    extract_and_transform_numeric_features,
    FeatureProcessing,
    get_bincode_features,
    get_capture_rows_by_ids,
    get_image,
    get_row_and_image_by_id,
    get_usage_instructions,
    jpeg,
    load_faiss_indices_from_lmdb,
    MAX_NN_VALUE,
)


def main():
    # Set layout to wide for optimal display of frog photos
    st.set_page_config(layout="wide", page_title="Pepeketua Interface")

    # Cache inference model for later
    get_inference_model()

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
                max_value=MAX_NN_VALUE,
                value=DEFAULT_KNN_VALUE,
            )

            # Scale factor for loading images
            scaling_options = sorted(jpeg.scaling_factors)
            scale_index = st.select_slider(
                label="Select scaling factor for image display",
                options=range(len(scaling_options)),
                value=1,
                format_func=lambda index: f"{scaling_options[index][0]}/{scaling_options[index][1]}",
            )
            scaling_factor = scaling_options[scale_index]
            submitted_knn = st.form_submit_button("Generate Images")

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

    nn_df = get_knn_results(grids, new_id_vectors, query_df, k_nearest)

    display_queries_and_nn_images(query_df, nn_df, query_image_bytes, scaling_factor)


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
        # Set the image bytes to the matched (single) index
        image_bytes.loc[image_index.idxmax()] = image_file.getvalue()

    # Reorder df to match image order and return relevant rows in df
    df = df.loc[image_bytes.index]
    df, image_bytes = df.reset_index(drop=True), image_bytes.reset_index(drop=True)

    return df, image_bytes


@st.experimental_memo
def calculate_query_id_vectors(
    df: pd.DataFrame, query_image_bytes: pd.Series
) -> Tuple[pd.Series, np.array, pd.Series]:
    # Load inference model and calculate query id vectors
    inference_model = get_inference_model()
    batch_df, query_image_bytes = inference_model.prepare_batch(df, query_image_bytes)
    new_id_vectors, _, grids = inference_model.predict(batch_df, query_image_bytes)
    return grids, new_id_vectors, query_image_bytes


@st.experimental_memo
def get_knn_results(
    grids: pd.Series,
    new_id_vectors: np.array,
    query_df: pd.DataFrame,
    k_nearest: int,
) -> pd.DataFrame:
    # nn_ids, distances_list, query_indices = [], [], []
    nn_ids, query_indices = [], []

    # Search for nearest neighbors to query vectors
    grid_faiss_indices = load_faiss_indices_from_lmdb()
    for grid_name, group in grids.groupby(grids):

        current_grid_query_indices = group.index
        query_vectors = new_id_vectors[current_grid_query_indices]

        grid_faiss_index = grid_faiss_indices[grid_name]
        distances, nn = grid_faiss_index.search(query_vectors, k=k_nearest)

        nn_ids.append(nn)
        query_indices.append(current_grid_query_indices)

    nn_ids = np.concatenate(nn_ids).astype(dtype=np.int32)
    query_indices = np.concatenate(query_indices)

    nn_ids = rerank_nn(nn_ids, query_df)

    nn_df = pd.DataFrame(
        data={"nn_ids": nn_ids.tolist()},
        index=query_indices,
    ).sort_index()

    return nn_df


def rerank_nn(nn_ids: np.array, query_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rerank the nearest neighbor results using the features defined in FeatureProcessing, if possible.
    :param nn_ids: Array of size len(query_df) x k_nearest, containing the nearest neighbor ids
    :param query_df: DataFrame containing the query data
    :return:
    """
    # If query_df does not contain all the columns required for feature extraction, return the original nn_ids
    if not all(col in query_df for col in FeatureProcessing.FEATURE_COLUMNS):
        return nn_ids

    def get_nn_features(ids: List[float]) -> np.array:
        capture_rows = get_capture_rows_by_ids(ids)
        numeric_features = extract_and_transform_numeric_features(capture_rows)
        bincode_features = get_bincode_features(capture_rows, is_query=False)
        return numeric_features, bincode_features

    nn_numeric_features, nn_bincode_features = list(
        zip(*map(get_nn_features, nn_ids.tolist()))
    )

    # Array of size len(query_df) x 1 x len(FeatureProcessing.FEATURE_COLUMNS) for broadcasting
    numeric_query_features = extract_and_transform_numeric_features(query_df)[
        :, np.newaxis, :
    ]
    # Array of size len(query_df) x k_nearest x len(FeatureProcessing.FEATURE_COLUMNS)
    nn_numeric_features = np.stack(nn_numeric_features)
    # Calculate the difference between the query features and the nearest neighbor features
    # Array of size len(query_df) x k_nearest
    numeric_diff = np.linalg.norm(
        numeric_query_features - nn_numeric_features, axis=2, keepdims=False
    )

    # Calculate bincode difference between each query and its nn (e.g. between "1100" and "1110"- the difference is 1)
    # Array of size len(query_df) x 1 for broadcasting
    query_bincode_features = get_bincode_features(query_df, is_query=True)[
        :, np.newaxis
    ]
    # Array of size len(query_df) x k_nearest
    nn_bincode_features = np.stack(nn_bincode_features)
    # Bincode diff of size len(query_df) x k_nearest
    bincode_diff = compare_bincodes(query_bincode_features, nn_bincode_features)

    # Add the bincode features difference to the numeric features difference
    diff = numeric_diff + bincode_diff

    # Sort (rerank) the nearest neighbor ids based on the difference
    reranking_index = np.argsort(diff, axis=1)
    nn_ids = np.take_along_axis(nn_ids, reranking_index, axis=1)
    return nn_ids


def display_queries_and_nn_images(
    query_df: pd.DataFrame,
    nn_df: pd.DataFrame,
    query_image_bytes: pd.Series,
    scaling_factor: Tuple[int, int],
):
    # Set up container structure to use to render nearest neighbor screen (main screen)
    (
        bottom_left,
        bottom_right,
        mid_left,
        mid_right,
        top_left,
        top_right,
        right_column_form,
    ) = generate_nn_screen_containers()

    # Left column showing the query images
    query_index = show_query_image_section(
        query_df,
        query_image_bytes,
        scaling_factor,
        bottom_left,
        mid_left,
        top_left,
    )

    # Right column showing nearest neighbor images
    show_nn_image_section(
        nn_df,
        query_index,
        scaling_factor,
        right_column_form,
        bottom_right,
        mid_right,
        top_right,
    )


def generate_nn_screen_containers() -> Tuple[
    st.container,
    st.container,
    st.container,
    st.container,
    st.container,
    st.container,
    st.form,
]:
    """Set up main nearest neighbor screen- two columns with three sections in each"""
    main_container = st.container()
    left_column, right_column = main_container.columns(2)

    right_column_form = right_column.form(key="nn form")

    top_right, mid_right, bottom_right = (
        right_column_form.empty(),
        right_column_form.empty(),
        right_column.container(),
    )
    top_left, mid_left, bottom_left = (
        left_column.empty(),
        left_column.empty(),
        left_column.container(),
    )
    return (
        bottom_left,
        bottom_right,
        mid_left,
        mid_right,
        top_left,
        top_right,
        right_column_form,
    )


def show_query_image_section(
    query_df: pd.DataFrame,
    query_image_bytes: pd.Series,
    scaling_factor: Tuple[int, int],
    bottom_left: st.container,
    mid_left: st.container,
    top_left: st.container,
) -> int:
    """Render left column showing the query images"""
    query_index = mid_left.selectbox(
        label="Select query to show",
        options=query_df.index,
        format_func=lambda q: query_df.loc[q, "filepath"],
    )
    top_left.image(
        get_image(query_image_bytes.loc[query_index], scaling_factor=scaling_factor),
    )
    bottom_left_expander = bottom_left.expander("Show query excel info")
    bottom_left_expander.dataframe(
        query_df.loc[query_index, :], use_container_width=True
    )
    return query_index


def show_nn_image_section(
    nn_df: pd.DataFrame,
    query_index: int,
    scaling_factor: Tuple[int, int],
    right_column_form: st.form,
    bottom_right: st.container,
    mid_right: st.container,
    top_right: st.container,
):
    """Showing nearest neighbor image section"""
    nn_indices = nn_df.loc[query_index, "nn_ids"]
    nn_number = mid_right.select_slider(
        label="Select nearest neighbor to view.",
        options=range(1, len(nn_indices) + 1),
        value=1,
    )
    nn_id = nn_indices[nn_number - 1]
    # The right column form submission button is used here
    # We ignore the button's result because we need the default rendered as well, so there won't be blank spaces.
    # Once "Show" is pressed, the image and data in expander are refreshed automatically.
    _ = right_column_form.form_submit_button(label="Show")
    show_nn_image_and_capture_info(nn_id, scaling_factor, bottom_right, top_right)


def show_nn_image_and_capture_info(
    nn_id: int,
    scaling_factor: Tuple[int, int],
    bottom_right: st.container,
    top_right: st.container,
):
    captured_frog_row, captured_frog_image_bytes = get_row_and_image_by_id(nn_id)
    top_right.image(get_image(captured_frog_image_bytes, scaling_factor=scaling_factor))
    bottom_right_expander = bottom_right.expander("Show capture info")
    bottom_right_expander.dataframe(captured_frog_row.T, use_container_width=True)


if __name__ == "__main__":
    main()
