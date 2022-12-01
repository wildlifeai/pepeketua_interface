import pandas as pd
import streamlit as st

if __name__ == "__main__":
    excel_file = st.file_uploader(label="Upload excel file with new rows")
    if excel_file:
        try:
            df = pd.read_excel(excel_file)
        except:
            st.write(
                f"{excel_file.name} isn't an excel file, please re upload the correct file."
            )

    image_files = st.file_uploader(
        label="Upload corresponding pictures", accept_multiple_files=True
    )
    # for
