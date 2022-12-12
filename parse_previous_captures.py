import datetime
import os
import zipfile

import numpy as np
import pandas as pd
from loguru import logger


def get_frog_picture_filepaths(frog_photo_dir: str) -> pd.DataFrame:

    """Load zip files"""
    # Load the five zipped files
    whareorino_a = zipfile.ZipFile(
        os.path.join(frog_photo_dir, "whareorino_a.zip"), "r"
    )
    whareorino_b = zipfile.ZipFile(
        os.path.join(frog_photo_dir, "whareorino_b.zip"), "r"
    )
    whareorino_c = zipfile.ZipFile(
        os.path.join(frog_photo_dir, "whareorino_c.zip"), "r"
    )
    whareorino_d = zipfile.ZipFile(
        os.path.join(frog_photo_dir, "whareorino_d.zip"), "r"
    )
    pukeokahu = zipfile.ZipFile(os.path.join(frog_photo_dir, "pukeokahu.zip"), "r")

    # Extract the filepath of the photos of individual frogs
    zips = [whareorino_a, whareorino_b, whareorino_c, whareorino_d, pukeokahu]
    pd_list = []

    for zip_file in zips:
        zip_pd = pd.DataFrame(
            [
                x
                for x in zip_file.namelist()
                if "Individual Frogs" in x and not x.endswith((".db", "/", "Store"))
            ]
        )
        pd_list.append(zip_pd)

    # Combine the file paths of the five grids into a single data frame
    frog_photo_file_list = pd.concat(pd_list).reset_index(drop=True)

    # Rename the column of df
    frog_photo_file_list = frog_photo_file_list.rename(columns={0: "filepath"})

    return frog_photo_file_list


def build_photo_file_list_df(frog_photo_dir: str) -> pd.DataFrame:
    """

    :param frog_photo_file_list:
    :return:
    """
    frog_photo_file_list = get_frog_picture_filepaths(frog_photo_dir)

    # Add new columns using directory and filename information
    expanded_path = frog_photo_file_list["filepath"].str.split("/", n=4, expand=True)

    # Add the grid, filename, and capture cols
    frog_photo_file_list["Grid"] = expanded_path[0]

    # frog_photo_file_list["frog_id"] = expanded_path[2]

    frog_photo_file_list["filename"] = expanded_path[3]

    frog_photo_file_list["Capture photo code"] = frog_photo_file_list[
        "filename"
    ].str.split(".", n=1, expand=True)[0]

    # Derive Capture # from file names
    # frog_photo_file_list["capture"] = (
    #     frog_photo_file_list["filename"]
    #     .str.split(".", n=1, expand=True)[0]  # get photo file basename
    #     .str.replace(
    #         "_", "-"
    #     )  # replace the underscores with dashes to get like 0101-738 and take right number
    #     .str.rsplit("-", n=1, expand=True)[
    #         1
    #     ]  # take the rightmost number before the dash, that's the capture id
    # )

    # Manually filter out non-standard photos
    # frog_photo_file_list = frog_photo_file_list[~frog_photo_file_list['filename'].str.contains(("Picture|IMG|#"))]

    return frog_photo_file_list


def load_excel_spreadsheets(
    pukeokahu_excel_file: str, whareorino_excel_file: str
) -> pd.DataFrame:
    """
    Load the excel spreadsheets
    Read the spreadsheets with frog capture information
    """
    whareorino_df = pd.read_excel(
        whareorino_excel_file,
        sheet_name=["Grid A", "Grid B", "Grid C", "Grid D"],
    )
    pukeokahu_df = pd.read_excel(
        pukeokahu_excel_file,
        sheet_name=["MR Data"],
    )
    """Add grid column to the frog capture info"""
    whareorino_df["Grid A"]["Grid"] = "Grid A"
    whareorino_df["Grid B"]["Grid"] = "Grid B"
    whareorino_df["Grid C"]["Grid"] = "Grid C"
    whareorino_df["Grid D"]["Grid"] = "Grid D"
    pukeokahu_df["MR Data"]["Grid"] = "Pukeokahu Frog Monitoring"

    # Combine datasets
    frog_id_df = pd.concat(
        [
            whareorino_df["Grid A"],
            whareorino_df["Grid B"],
            whareorino_df["Grid C"],
            whareorino_df["Grid D"],
            pukeokahu_df["MR Data"],
        ]
    ).reset_index(drop=True)
    return frog_id_df, whareorino_df, pukeokahu_df


def check_column_consistency(
    pukeokahu_df: pd.DataFrame, whareorino_df: pd.DataFrame
) -> None:
    """Check for consistent column names"""

    # AB
    col_diff = list(
        set(whareorino_df["Grid A"].columns) - set(whareorino_df["Grid B"].columns)
    )
    if col_diff:
        logger.info("Differences between A and B", col_diff)

    # BA
    col_diff = list(
        set(whareorino_df["Grid B"].columns) - set(whareorino_df["Grid A"].columns)
    )
    if col_diff:
        logger.info("Differences between B and A", col_diff)

    # AC
    col_diff = list(
        set(whareorino_df["Grid A"].columns) - set(whareorino_df["Grid C"].columns)
    )
    if col_diff:
        logger.info("Differences between A and C", col_diff)

    # CA
    col_diff = list(
        set(whareorino_df["Grid C"].columns) - set(whareorino_df["Grid A"].columns)
    )
    if col_diff:
        logger.info("Differences between C and A", col_diff)

    # AD
    col_diff = list(
        set(whareorino_df["Grid A"].columns) - set(whareorino_df["Grid D"].columns)
    )
    if col_diff:
        logger.info("Differences between A and D", col_diff)

    # DA
    col_diff = list(
        set(whareorino_df["Grid D"].columns) - set(whareorino_df["Grid A"].columns)
    )
    if col_diff:
        logger.info("Differences between D and A", col_diff)

    # AP
    col_diff = list(
        set(whareorino_df["Grid A"].columns) - set(pukeokahu_df["MR Data"].columns)
    )
    if col_diff:
        logger.info("Differences between A and pukeokahu", col_diff)

    # PA
    col_diff = list(
        set(pukeokahu_df["MR Data"].columns) - set(whareorino_df["Grid A"].columns)
    )
    if col_diff:
        logger.info("Differences between pukeokahu and A", col_diff)


def filter_entries_from_time(frog_id_df):
    """Limit the df to frog identifications older than 2020"""
    # Select rows with valid dates
    valid_frog_id_df = frog_id_df[
        (frog_id_df["Date"].notnull()) & (frog_id_df["Date"] != "Date")
    ]

    # Filter observations older than 2020
    valid_frog_id_df = valid_frog_id_df[
        valid_frog_id_df["Date"].astype("datetime64[ns]")
        < datetime.datetime(year=2020, month=1, day=1)
    ]
    return valid_frog_id_df


def filter_faulty_entries(frog_id_df):
    """Remove manual typos and faulty entries"""
    wrong_capture_id = ["GRID SEARCHED BUT ZERO FROGS FOUND =(", "hochstetter"]
    frog_id_df = frog_id_df[~frog_id_df["Capture #"].isin(wrong_capture_id)]
    # Remove empty capture
    frog_id_df = frog_id_df.dropna(subset=["Capture #"])
    # Remove empty capture
    frog_id_df = frog_id_df.dropna(subset=["Capture photo code"])
    # Number of photos identified per grid
    frog_id_df.groupby(["Grid"])["Grid"].count()
    return frog_id_df


def check_duplicated_photos(merged_frog_id_filename: pd.DataFrame) -> None:
    """Find out duplicated photos"""
    if (
        merged_frog_id_filename[
            merged_frog_id_filename.duplicated(
                ["Capture photo code", "Grid"], keep=False
            )
        ][["Capture #", "Grid", "Capture photo code", "filepath"]].shape[0]
        > 0
    ):
        logger.info(
            (
                "There are",
                merged_frog_id_filename[
                    merged_frog_id_filename.duplicated(
                        ["Capture photo code", "Grid"], keep=False
                    )
                ][["Capture #", "Grid", "Capture photo code", "filepath"]].shape[0],
                "duplicates",
            )
        )
        logger.info(
            merged_frog_id_filename[
                merged_frog_id_filename.duplicated(
                    ["Capture photo code", "Grid"], keep=False
                )
            ][["Capture #", "Grid", "Capture photo code", "filepath"]]
        )
        logger.info(
            "Saving list of duplicated frog photos to duplicated_frog_photos.csv"
        )
        merged_frog_id_filename[
            merged_frog_id_filename.duplicated(
                ["Capture photo code", "Grid"], keep=False
            )
        ][["Capture #", "Grid", "Capture photo code", "filepath"]].to_csv(
            "duplicated_frog_photos.csv"
        )


def try_to_eliminate_filepath_nans(
    frog_photo_file_list: pd.DataFrame, merged_frog_id_filename: pd.DataFrame
) -> pd.DataFrame:
    """Find out identifications that can't be mapped to a photo (missing filepaths)"""
    # Missing filepaths per grid
    logger.info("Number of missing filepaths by grid:")
    logger.info(
        merged_frog_id_filename[merged_frog_id_filename.columns.difference(["Grid"])]
        .isnull()
        .groupby(merged_frog_id_filename["Grid"])
        .sum()
        .astype(int)["filepath"]
    )
    # Rename original photo code
    merged_frog_id_filename = merged_frog_id_filename.rename(
        columns={"Capture photo code": "Original Capture photo code"}
    )
    # Modify 'Capture photo code' using the marks and Capture # of those photos unable to be located
    merged_frog_id_filename["Capture photo code"] = np.where(
        merged_frog_id_filename["filepath"].isna(),
        merged_frog_id_filename["Back left mark"]
        .astype(str)
        .apply(lambda x: "_" if "?" in x else x)
        + merged_frog_id_filename["Back right mark"]
        .astype(str)
        .apply(lambda x: "_" if "?" in x else x)
        + merged_frog_id_filename["Face left mark"]
        .astype(str)
        .apply(lambda x: "_" if "?" in x else x)
        + merged_frog_id_filename["Face right mark"]
        .astype(str)
        .apply(lambda x: "_" if "?" in x else x)
        + "-"
        + merged_frog_id_filename["Capture #"].astype(int).astype(str),
        merged_frog_id_filename["Original Capture photo code"],
    )
    # Add filename and filepath of the photos to the frog identification dataframe again
    # with the updated 'Capture photo code'
    logger.info(
        "Adding filename and filepath of the photos to the frog identification dataframe again "
        "with the updated 'Capture photo code'"
    )
    merged_frog_id_filename = merged_frog_id_filename.drop(
        columns=frog_photo_file_list.columns.difference(["Capture photo code", "Grid"])
    ).merge(frog_photo_file_list, on=["Capture photo code", "Grid"], how="left")
    logger.info("Number of null filepaths now:")
    logger.info(
        merged_frog_id_filename[merged_frog_id_filename.columns.difference(["Grid"])]
        .isnull()
        .groupby("Grid")
        .sum()
        .astype(int)["filepath"]
    )
    # Modify 'Capture photo code' using the marks and Capture # of those photos unable to be located
    merged_frog_id_filename["Capture photo code"] = np.where(
        merged_frog_id_filename["filepath"].isna(),
        merged_frog_id_filename["Back left mark"]
        .astype(str)
        .apply(lambda x: "0" if "?" in x else x)
        + merged_frog_id_filename["Back right mark"]
        .astype(str)
        .apply(lambda x: "0" if "?" in x else x)
        + merged_frog_id_filename["Face left mark"]
        .astype(str)
        .apply(lambda x: "0" if "?" in x else x)
        + merged_frog_id_filename["Face right mark"]
        .astype(str)
        .apply(lambda x: "0" if "?" in x else x)
        + "-"
        + merged_frog_id_filename["Capture #"].astype(int).astype(str),
        merged_frog_id_filename["Capture photo code"],
    )
    # Add filepath of the photos to each frog identification again with the updated 'Capture photo code'
    merged_frog_id_filename = merged_frog_id_filename.drop(
        columns=frog_photo_file_list.columns.difference(["Capture photo code", "Grid"])
    ).merge(frog_photo_file_list, on=["Capture photo code", "Grid"], how="left")
    logger.info(
        merged_frog_id_filename[merged_frog_id_filename.columns.difference(["Grid"])]
        .isnull()
        .groupby(merged_frog_id_filename["Grid"])
        .sum()
        .astype(int)["filepath"]
    )
    # Modify 'Capture photo code' using the marks and Capture # of those photos unable to be located
    merged_frog_id_filename["Capture photo code"] = np.where(
        merged_frog_id_filename["filepath"].isna(),
        merged_frog_id_filename["Back left mark"]
        .astype(str)
        .apply(lambda x: "1" if "?" in x else x)
        + merged_frog_id_filename["Back right mark"]
        .astype(str)
        .apply(lambda x: "1" if "?" in x else x)
        + merged_frog_id_filename["Face left mark"]
        .astype(str)
        .apply(lambda x: "1" if "?" in x else x)
        + merged_frog_id_filename["Face right mark"]
        .astype(str)
        .apply(lambda x: "1" if "?" in x else x)
        + "-"
        + merged_frog_id_filename["Capture #"].astype(int).astype(str),
        merged_frog_id_filename["Capture photo code"],
    )
    # Add filepath of the photos to each frog identification again with the updated 'Capture photo code'
    merged_frog_id_filename = merged_frog_id_filename.drop(
        columns=list(
            set(list(frog_photo_file_list.columns))
            - set(["Capture photo code", "Grid"])
        )
    ).merge(frog_photo_file_list, on=["Capture photo code", "Grid"], how="left")
    logger.info(
        merged_frog_id_filename[merged_frog_id_filename.columns.difference(["Grid"])]
        .isnull()
        .groupby(merged_frog_id_filename["Grid"])
        .sum()
        .astype(int)["filepath"]
    )
    merged_frog_id_filename = merged_frog_id_filename.rename(
        columns={"Capture photo code": "updated Capture photo code"}
    )
    return merged_frog_id_filename


def main(frog_photo_dir: str, whareorino_excel_file: str, pukeokahu_excel_file: str):
    """

    :param frog_photo_dir:
    :param whareorino_excel_file:
    :param pukeokahu_excel_file:
    :return:
    """

    """Prepare information related to the photos"""

    frog_photo_file_list = build_photo_file_list_df(frog_photo_dir)

    frog_id_df, whareorino_df, pukeokahu_df = load_excel_spreadsheets(
        pukeokahu_excel_file, whareorino_excel_file
    )

    # filter entries later than 1.1.2020
    frog_id_df = filter_entries_from_time(frog_id_df)

    # filter typos and bad entries
    frog_id_df = filter_faulty_entries(frog_id_df)

    """Match the frog identification data to the photo file paths data"""
    merged_frog_id_filename = frog_id_df.merge(
        frog_photo_file_list, on=["Capture photo code", "Grid"], how="left"
    )

    """
    Work in progress to clean and tidy out the data
    """
    check_duplicated_photos(merged_frog_id_filename)

    merged_frog_id_filename = try_to_eliminate_filepath_nans(
        frog_photo_file_list, merged_frog_id_filename
    )

    merged_frog_id_filename = merged_frog_id_filename.drop_duplicates(
        ["Capture #", "Grid"]
    )

    new_df = frog_id_df.merge(
        merged_frog_id_filename[["Capture #", "Grid", "updated Capture photo code"]],
        on=["Capture #", "Grid"],
        how="left",
    )

    # make sure if empty original values are used
    new_df["updated Capture photo code"] = np.where(
        new_df["updated Capture photo code"].isna(),
        new_df["Capture photo code"],
        new_df["updated Capture photo code"],
    )

    new_df["different Capture photo Code"] = np.where(
        new_df["Capture photo code"] == new_df["updated Capture photo code"], 0, 1
    )

    # Closest match between the Capture photo code and filenames
    new_df[new_df["Grid"] == "Grid A"].drop(columns=["Grid"]).to_csv(
        "victor_reviewed_grid_a.csv"
    )
    new_df[new_df["Grid"] == "Grid B"].drop(columns=["Grid"]).to_csv(
        "victor_reviewed_grid_b.csv"
    )
    new_df[new_df["Grid"] == "Grid C"].drop(columns=["Grid"]).to_csv(
        "victor_reviewed_grid_c.csv"
    )
    new_df[new_df["Grid"] == "Grid D"].drop(columns=["Grid"]).to_csv(
        "victor_reviewed_grid_d.csv"
    )
    new_df[new_df["Grid"] == "Pukeokahu Frog Monitoring"].drop(columns=["Grid"]).to_csv(
        "victor_reviewed_pukeokahu.csv"
    )

    # Missing photos
    merged_frog_id_filename[
        (merged_frog_id_filename["Grid"] == "Grid A")
        & (merged_frog_id_filename["filepath"].isna())
    ].to_csv("missing_grid_a.csv")
    merged_frog_id_filename[
        (merged_frog_id_filename["Grid"] == "Grid B")
        & (merged_frog_id_filename["filepath"].isna())
    ].to_csv("missing_grid_b.csv")
    merged_frog_id_filename[
        (merged_frog_id_filename["Grid"] == "Grid C")
        & (merged_frog_id_filename["filepath"].isna())
    ].to_csv("missing_grid_c.csv")
    merged_frog_id_filename[
        (merged_frog_id_filename["Grid"] == "Grid D")
        & (merged_frog_id_filename["filepath"].isna())
    ].to_csv("missing_grid_d.csv")
    merged_frog_id_filename[
        (merged_frog_id_filename["Grid"] == "Pukeokahu Frog Monitoring")
        & (merged_frog_id_filename["filepath"].isna())
    ].to_csv("missing_pukeokahu.csv")

    # Add filepath info for whareorino_df
    whareorino_df_a_complete_df = whareorino_df["Grid A"].merge(
        frog_photo_file_list, on=["Capture photo code", "Grid"], how="left"
    )

    whareorino_df_b_complete_df = whareorino_df["Grid B"].merge(
        frog_photo_file_list, on=["Capture photo code", "Grid"], how="left"
    )

    whareorino_df_c_complete_df = whareorino_df["Grid C"].merge(
        frog_photo_file_list, on=["Capture photo code", "Grid"], how="left"
    )

    whareorino_df_d_complete_df = whareorino_df["Grid D"].merge(
        frog_photo_file_list, on=["Capture photo code", "Grid"], how="left"
    )

    # Add filepath info for pukeokahu
    pukeokahu_complete_df = pukeokahu_df["MR Data"].merge(
        frog_photo_file_list, on=["Capture photo code", "Grid"], how="left"
    )


if __name__ == "__main__":
    # Log to disk
    logger.add("parse_previous_captures.log")

    frog_photo_dir = "/Users/lioruzan/Downloads/frog_photos"
    whareorino_excel_file = "/Users/lioruzan/Downloads/Whareorino frog monitoring data 2005 onwards CURRENT FILE - DOCDM-106978.xls"
    pukeokahu_excel_file = "/Users/lioruzan/Downloads/Pukeokahu Monitoring Data 2006 onwards - DOCDM-95563.xls"
    main(frog_photo_dir, whareorino_excel_file, pukeokahu_excel_file)
