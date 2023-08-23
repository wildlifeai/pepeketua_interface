import datetime
import pickle
from os.path import join
from typing import Dict, List, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
import sqlalchemy as db
from loguru import logger
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utilities.lmdb_classes import LmdbWriter
from utilities.utilities import (
    extract_features,
    FILES_PATH,
    LMDB_PATH,
    PUKEOKAHU_EXCEL_FILE,
    SQL_SERVER_STRING,
    WHAREORINO_EXCEL_FILE,
    ZIP_NAMES,
    ZIP_PATH
)

"""
The whole point of this file is to parse old frog sightings (before 2020) 
and match each row to it's picture in Individual Frogs dir
"""


def get_frog_photo_filepaths(photo_dir: str, zips: List[str]) -> pd.DataFrame:
    """Load zip files that contain individual frog photo list"""
    zip_photo_lists = []
    for zip_name in zips:
        with ZipFile(join(photo_dir, zip_name), mode="r") as zip_file:
            zip_photo_list = pd.DataFrame(
                {
                    "filepath": [
                        x
                        for x in zip_file.namelist()
                        if "Individual Frogs" in x
                        and not x.endswith((".db", "/", "Store"))
                    ],
                    "zip_source": zip_name,
                },
            )
            zip_photo_lists.append(zip_photo_list)
    # Combine the file paths of the five grids into a single data frame
    frog_photo_list = pd.concat(zip_photo_lists).reset_index(drop=True)

    return frog_photo_list


def expand_photo_file_list_df(photo_dir: str, zips: List[str]) -> pd.DataFrame:
    """

    :param photo_dir:
    :param zips:
    :return:
    """
    frog_photo_list = get_frog_photo_filepaths(photo_dir, zips)

    # Add new columns using directory and filename information
    expanded_path = frog_photo_list["filepath"].str.split("/", n=4, expand=True)

    # Add the grid, filename, and capture cols
    frog_photo_list["Grid"] = expanded_path[0]

    frog_photo_list["folder_frog_id"] = expanded_path[2]

    frog_photo_list["Capture photo code"] = expanded_path[3].str.split(
        ".", n=1, expand=True
    )[0]

    return frog_photo_list


def load_excel_spreadsheets(
    pukeokahu_excel_file: str, whareorino_excel_file: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def try_to_eliminate_filepath_nans(
    frog_photo_file_list: pd.DataFrame, merged_frog_id_filepath: pd.DataFrame
) -> pd.DataFrame:
    """
    Try to find intentifications that can't be mapped to a photo (missing filepaths) by the following method:
    Replace original Capture photo code with manually generated photo codes.
    For example, for rows where filepath is nan, transform capture photo code 01?11-888 -> 0_11/0111/0011-888
    then re-merge with frog_photo_file_list on left on Capture photo code.

    This new capture photo code might match a file on disk that was previously unmatched, thus filling in the filepath.
    :param frog_photo_file_list:
    :param merged_frog_id_filepath:
    :return:
    """

    # Missing filepaths per grid
    logger.info("Number of missing filepaths by grid:")
    logger.info(
        merged_frog_id_filepath[merged_frog_id_filepath.columns.difference(["Grid"])]
        .isnull()
        .groupby(merged_frog_id_filepath["Grid"])
        .sum()
        .astype(int)["filepath"]
    )

    # Back up original capture photo code:
    merged_frog_id_filepath["Original capture photo code"] = merged_frog_id_filepath[
        "Capture photo code"
    ].copy()

    # Trying to recover filepaths by changing Capture photo code
    logger.info("Rewriting 'Capture photo code' in rows where filpath is nan.")

    merged_frog_id_filepath = (
        fill_filepath_nans_by_replacing_question_mark_in_capture_photo_code(
            frog_photo_file_list, merged_frog_id_filepath, replacement="_"
        )
    )

    merged_frog_id_filepath = (
        fill_filepath_nans_by_replacing_question_mark_in_capture_photo_code(
            frog_photo_file_list, merged_frog_id_filepath, replacement="0"
        )
    )

    merged_frog_id_filepath = (
        fill_filepath_nans_by_replacing_question_mark_in_capture_photo_code(
            frog_photo_file_list, merged_frog_id_filepath, replacement="1"
        )
    )

    # Set aside the updated capture photo code
    merged_frog_id_filepath = merged_frog_id_filepath.rename(
        columns={"Capture photo code": "Updated capture photo code"}
    )

    # Restore original values where filepath remained null, otherwise take updated capture photo codes
    merged_frog_id_filepath["Capture photo code"] = np.where(
        merged_frog_id_filepath["filepath"].isna(),
        merged_frog_id_filepath["Original capture photo code"],
        merged_frog_id_filepath["Updated capture photo code"],
    )

    # Make sure there's no nans in 'Capture photo code'
    assert bool(
        merged_frog_id_filepath["Capture photo code"].notna().all()
    ), "Found nans in 'Capture photo code', something went wrong."

    # Log difference between old and new capture photo codes
    same_capture_code_mask = (
        merged_frog_id_filepath["Capture photo code"]
        == merged_frog_id_filepath["Updated capture photo code"]
    )
    merged_frog_id_filepath["Different capture photo code"] = np.where(
        same_capture_code_mask,
        False,
        True,
    )

    missing_filepaths = merged_frog_id_filepath["filepath"].isna()
    logger.info(f"Total number of missing filepaths: {sum(missing_filepaths)}")

    # Save the list of rows with missing photos
    merged_frog_id_filepath[missing_filepaths].to_csv(
        join(FILES_PATH, "missing_photos.csv")
    )

    return merged_frog_id_filepath


def fill_filepath_nans_by_replacing_question_mark_in_capture_photo_code(
    frog_photo_file_list: pd.DataFrame,
    merged_frog_id_filepath: pd.DataFrame,
    replacement: str,
) -> pd.DataFrame:
    # Modify 'Capture photo code' using the marks and Capture # of those photos unable to be located

    logger.info(f"Replacing '?' with '{replacement}' then re-merging.")

    new_capture_photo_code = (
        merged_frog_id_filepath["Back left mark"]
        .astype(str)
        .apply(lambda x: replacement if "?" in x else x)
        + merged_frog_id_filepath["Back right mark"]
        .astype(str)
        .apply(lambda x: replacement if "?" in x else x)
        + merged_frog_id_filepath["Face left mark"]
        .astype(str)
        .apply(lambda x: replacement if "?" in x else x)
        + merged_frog_id_filepath["Face right mark"]
        .astype(str)
        .apply(lambda x: replacement if "?" in x else x)
        + "-"
        + merged_frog_id_filepath["Capture #"].astype(int).astype(str)
    )

    merged_frog_id_filepath["Capture photo code"] = np.where(
        merged_frog_id_filepath["filepath"].isna(),
        new_capture_photo_code,
        merged_frog_id_filepath["Capture photo code"],
    )

    # Drop filename\filepath cols then re-merge over new Capture photo code and Grid
    merged_frog_id_filepath = (
        merged_frog_id_filepath.drop(
            columns=frog_photo_file_list.columns.difference(
                ["Capture photo code", "Grid"]
            )
        )
        .drop_duplicates()
        .merge(frog_photo_file_list, on=["Capture photo code", "Grid"], how="left")
        .reset_index(drop=True)
    )

    logger.info("Current number of missing filepath by grid:")
    logger.info(
        merged_frog_id_filepath[merged_frog_id_filepath.columns.difference(["Grid"])]
        .isnull()
        .groupby(merged_frog_id_filepath["Grid"])
        .sum()
        .astype(int)["filepath"]
    )
    return merged_frog_id_filepath


def find_incorrect_filepaths(
    merged_frog_id_filepath: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare numbers in frog ID # to the same numbers obtained from "folder_frog_id" column:
    <Grid>/Individual Frogs/<folder_frog_id>/<Capture photo code>.jpg
    folder_frog_id matches number in Frog ID # format: A5, C1000 for example.
    Rows that don't match could be a misplaced file or a bad capture photo code
    :param merged_frog_id_filepath:
    :return:
    """
    # Use regex to isolate the number in the Frog ID #
    # Then compare the results with "folder_frog_id"
    frog_id_num = (
        merged_frog_id_filepath["Frog ID #"]
        .astype("string")
        .str.extract(r"(?P<frog_id_num>\d+)", expand=False)
    )
    folder_frog_id = merged_frog_id_filepath["folder_frog_id"].astype("string")
    merged_frog_id_filepath["do_frog_ids_match"] = frog_id_num == folder_frog_id

    # Save rows where Frog ID # and folder_frog_id have different numbers
    false_matches = merged_frog_id_filepath[
        merged_frog_id_filepath["do_frog_ids_match"] == False
    ]

    logger.info(f"There are {len(false_matches)} rows with mismatched filepath.")
    false_matches.to_csv(join(FILES_PATH, "incorrect_filepaths.csv"))

    return merged_frog_id_filepath


def save_photos_to_lmdb(df: pd.DataFrame, zip_path: str) -> pd.DataFrame:
    """
    Save photos straight from zip to lmdb
    :param df:
    :param zip_path: basedir of zip files (beside this folder the lmdb dir will be created)
    :return:
    """

    def write_photo(row: Dict) -> None:
        if pd.isna(row["lmdb_key"]):
            return
        with ZipFile(join(zip_path, row["zip_source"]), mode="r") as zip_file:
            writer.add(row["lmdb_key"].encode(), zip_file.read(row["filepath"]))

    # Set lmdb key to be the index as string in the format "000000012" if filepath exists, nan otherwise
    df["lmdb_key"] = df["id"].apply(lambda ix: f"{ix:09}")
    df["lmdb_key"] = df["lmdb_key"].where(df["filepath"].notna(), np.nan)

    # Write photos to lmdb sequentially
    logger.info("Writing photos to lmdb.")
    with LmdbWriter(output_path=LMDB_PATH) as writer:
        for row in tqdm(df.to_dict("records")):
            write_photo(row)

    return df


def fit_and_save_scaler(merged_frog_id_filepath_df: pd.DataFrame) -> None:
    """Fit and save StandardScaler to lmdb for later result reranking"""
    scaler = StandardScaler()
    svl_weight = extract_features(merged_frog_id_filepath_df)

    # Rows which contain NA are not used to compute the mean and std
    scaler.fit(svl_weight)

    # Save for later use
    with LmdbWriter(output_path=LMDB_PATH) as writer:
        writer.add(b"scaler", pickle.dumps(scaler))


def save_to_postgres(df: pd.DataFrame, sql_server_string: str) -> None:
    """Save all frog information up until now to the local PostgreSQL server"""
    engine = db.create_engine(sql_server_string)

    with engine.connect() as con:
        df.to_sql("frogs", con, if_exists="replace", index=False)

    engine.dispose()


def run():
    # Log to disk
    logger.add(join(FILES_PATH, "parse_previous_captures.log"))

    """Prepare information related to the photos"""

    frog_photo_file_list_df = expand_photo_file_list_df(ZIP_PATH, ZIP_NAMES)

    frog_id_df, whareorino_df, pukeokahu_df = load_excel_spreadsheets(
        PUKEOKAHU_EXCEL_FILE, WHAREORINO_EXCEL_FILE
    )

    # filter entries later than 1.1.2020
    frog_id_df = filter_entries_from_time(frog_id_df)

    # filter typos and bad entries
    frog_id_df = filter_faulty_entries(frog_id_df)

    """Match the frog identification data to the photo file paths data"""
    merged_frog_id_filepath_df = frog_id_df.merge(
        frog_photo_file_list_df, on=["Capture photo code", "Grid"], how="left"
    ).reset_index(drop=True)

    """
    Work in progress to clean and tidy out the data
    """
    merged_frog_id_filepath_df = try_to_eliminate_filepath_nans(
        frog_photo_file_list_df, merged_frog_id_filepath_df
    )

    merged_frog_id_filepath_df = find_incorrect_filepaths(merged_frog_id_filepath_df)

    # Add index as "id" for future reference
    merged_frog_id_filepath_df.loc[
        :, "id"
    ] = merged_frog_id_filepath_df.index.to_series()

    merged_frog_id_filepath_df = save_photos_to_lmdb(
        merged_frog_id_filepath_df, ZIP_PATH
    )

    # Fit a StandardScaler to the SVL and Weight columns for later use
    fit_and_save_scaler(merged_frog_id_filepath_df)

    save_to_postgres(merged_frog_id_filepath_df, SQL_SERVER_STRING)


if __name__ == "__main__":
    run()
