import pandas as pd
import streamlit as st

from capture_processing.inference_model import get_inference_model
from utilities.utilities import (
    display_image_and_k_nn,
    K_NEAREST,
    load_indices_from_lmdb,
    prepare_batch,
)

if __name__ == "__main__":
    st.write("## Upload your excel with new observations.")
    excel_file = st.file_uploader(label="Excel uploader", label_visibility="hidden")
    if excel_file:

        st.write("### Enter sheet name or select 'Use first sheet' checkbox.")
        sheet_name = st.text_input(label="Sheet name", label_visibility="hidden")

        if st.checkbox(label="Use first sheet"):
            sheet_name = 0

        if sheet_name != "":
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
            except Exception as e:
                st.write(
                    f"Error loading {excel_file.name}. Please correct the file or the sheet name."
                )
                raise e

            st.write(
                f"###### Loaded sheet '{sheet_name}' from {excel_file.name}."
                if sheet_name != 0
                else f"###### Loaded the first sheet from {excel_file.name}."
            )

            st.write("## Upload corresponding pictures.")
            image_files = st.file_uploader(
                label="Picture uploader",
                label_visibility="hidden",
                accept_multiple_files=True,
            )
            st.write("###### Press this button when finished uploading the images.")
            if st.button(label="Press here"):
                if image_files:

                    assert (
                        "filepath" in df
                    ), "Excel sheet must contain the column 'filepath', please re-upload the correct sheet"

                    for image_file in image_files:
                        image_index = df["filepath"].str.endswith(
                            image_file.name, na=False
                        )

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

                        df.loc[image_index, "image_bytes"] = image_file.getvalue()

                    # Collect and show results
                    image_names = tuple(image_file.name for image_file in image_files)
                    image_indices = df["filepath"].str.endswith(image_names, na=False)
                    if len(df[image_indices]) != len(df):
                        imageless_rows = df[~image_indices]
                        st.write(
                            f"{len(imageless_rows)} rows lack images: ",
                            imageless_rows.head(),
                        )

                    # Get image contents and start process of identity vector collection
                    batch_df = df[image_indices]
                    batch_df = prepare_batch(batch_df)
                    inference_model = get_inference_model()
                    new_id_vectors, _, grids = inference_model.predict(batch_df)

                    indices = load_indices_from_lmdb()

                    grid_groups = grids.groupby(grids).groups
                    for grid, group in grid_groups.items():
                        index = indices[grid]
                        query_vectors = new_id_vectors[group]
                        _, nn = index.search(query_vectors, k=K_NEAREST)
                        images_bytes = batch_df.loc[group, "image_bytes"]
                        for new_frog_num, image_bytes, k_nn in zip(
                            group, images_bytes, nn
                        ):
                            display_image_and_k_nn(
                                new_frog_num, image_bytes, k_nn.tolist()
                            )
