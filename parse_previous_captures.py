import zipfile

import numpy as np
import pandas as pd

"""

"""


def enumerate_frog_pics(frog_photo_dir: str) -> pd.DataFrame:

    """Load zip files"""
    # Load the five zipped files
    whareorino_a = zipfile.ZipFile(frog_photo_dir + "whareorino_a.zip", "r")
    whareorino_b = zipfile.ZipFile(frog_photo_dir + "whareorino_b.zip", "r")
    whareorino_c = zipfile.ZipFile(frog_photo_dir + "whareorino_c.zip", "r")
    whareorino_d = zipfile.ZipFile(frog_photo_dir + "whareorino_d.zip", "r")
    pukeokahu = zipfile.ZipFile(frog_photo_dir + "pukeokahu.zip", "r")

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
    frog_photo_list = pd.concat(pd_list)

    # Rename the column of df
    frog_photo_list = frog_photo_list.rename(columns={0: "filepath"})

    return frog_photo_list


def main(frog_photo_dir: str, whareorino_excel_file: str, pukeokahu_excel_file: str):

    """Prepare information related to the photos"""
    frog_df = enumerate_frog_pics(frog_photo_dir)

    # Add new columns using directory and filename information
    directories = frog_df["filepath"].str.split("/", n=4, expand=True)

    # Add the grid, frog_id, filename, and capture cols
    frog_df["grid"] = directories[0]
    frog_df["frog_id"] = directories[2]
    frog_df["filename"] = directories[3]
    frog_df["Capture photo code"] = frog_df["filename"].str.split(".", 1, expand=True)[
        0
    ]
    frog_df["capture"] = (
        frog_df["filename"]
        .str.split(".", 1, expand=True)[0]
        .str.replace("_", "-")
        .str.rsplit("-", 1, expand=True)[1]
    )

    # Manually filter out non-standard photos
    # frog_df = frog_df[~frog_df['filename'].str.contains(("Picture|IMG|#"))]

    """
    Load the excel spreadsheets
    Read the spreadsheets with frog capture information
    """
    whareorino_df = pd.read_excel(
        "/content/drive/MyDrive/Projects/pepeketua_id/Whareorino frog monitoring data 2005 onwards CURRENT FILE - DOCDM-106978.xls",
        sheet_name=["Grid A", "Grid B", "Grid C", "Grid D"],
    )
    pukeokahu_df = pd.read_excel(
        "/content/drive/MyDrive/Projects/pepeketua_id/Pukeokahu Monitoring Data 2006 onwards - DOCDM-95563.xls",
        sheet_name=["MR Data"],
    )

    """Add grid column to the frog capture info"""

    whareorino_df["Grid A"]["grid"] = "Grid A"
    whareorino_df["Grid B"]["grid"] = "Grid B"
    whareorino_df["Grid C"]["grid"] = "Grid C"
    whareorino_df["Grid D"]["grid"] = "Grid D"
    pukeokahu_df["MR Data"]["grid"] = "Pukeokahu Frog Monitoring"

    # Combine datasets
    frog_id_df = pd.concat(
        [
            whareorino_df["Grid A"],
            whareorino_df["Grid B"],
            whareorino_df["Grid C"],
            whareorino_df["Grid D"],
            pukeokahu_df["MR Data"],
        ]
    )
    """Limit the df to frog identifications older than 2020"""
    # Select rows with valid dates
    valid_frog_id_df = frog_id_df[
        (frog_id_df["Date"].notnull()) & (frog_id_df["Date"] != "Date")
    ]

    # Filter observations older than 2020
    valid_frog_id_df = valid_frog_id_df[
        valid_frog_id_df["Date"].astype("datetime64[ns]")
    ]

    """Remove manual typos and faulty entries"""

    wrong_capture_id = ["GRID SEARCHED BUT ZERO FROGS FOUND =(", "hochstetter"]
    valid_frog_id_df = valid_frog_id_df[
        ~valid_frog_id_df["Capture #"].isin(wrong_capture_id)
    ]

    # Remove empty capture
    valid_frog_id_df = valid_frog_id_df.dropna(subset=["Capture #"])

    # Remove empty capture
    valid_frog_id_df = valid_frog_id_df.dropna(subset=["Capture photo code"])

    # Number of photos identified per grid
    valid_frog_id_df.groupby(["grid"])["grid"].count()

    """Map the photos with the frog identification data"""
    df = valid_frog_id_df.merge(frog_df, on=["Capture photo code", "grid"], how="left")

    print(df.groupby(["grid"])["grid"].count())

    """
    Work in progress to clean and tidy out the data
    """

    """Find out duplicated photos"""
    if (
        df[df.duplicated(["Capture photo code", "grid"], keep=False)][
            ["Capture #", "grid", "Capture photo code", "filepath"]
        ].shape[0]
        > 0
    ):
        print(
            "There are",
            df[df.duplicated(["Capture photo code", "grid"], keep=False)][
                ["Capture #", "grid", "Capture photo code", "filepath"]
            ].shape[0],
            "duplicates",
        )
        print(
            df[df.duplicated(["Capture photo code", "grid"], keep=False)][
                ["Capture #", "grid", "Capture photo code", "filepath"]
            ]
        )
        df[df.duplicated(["Capture photo code", "grid"], keep=False)][
            ["Capture #", "grid", "Capture photo code", "filepath"]
        ].to_csv("duplicated_frog_photos.csv")

    """Find out identifications that can't be mapped to a photo (missing filepaths)"""

    # Missing filepaths per grid
    print(
        df[df.columns.difference(["grid"])]
        .isnull()
        .groupby(df.grid)
        .sum()
        .astype(int)["filepath"]
    )

    # Rename original photo code
    df = df.rename(columns={"Capture photo code": "Original Capture photo code"})

    # Modify 'Capture photo code' using the marks and Capture # of those photos unable to be located
    df["Capture photo code"] = np.where(
        df["filepath"].isna(),
        df["Back left mark"].astype(str).apply(lambda x: "_" if "?" in x else x)
        + df["Back right mark"].astype(str).apply(lambda x: "_" if "?" in x else x)
        + df["Face left mark"].astype(str).apply(lambda x: "_" if "?" in x else x)
        + df["Face right mark"].astype(str).apply(lambda x: "_" if "?" in x else x)
        + "-"
        + df["Capture #"].astype(int).astype(str),
        df["Original Capture photo code"],
    )

    # Add filepath of the photos to each frog identification again with the updated 'Capture photo code'
    df = df.drop(
        columns=list(set(list(frog_df.columns)) - set(["Capture photo code", "grid"]))
    ).merge(frog_df, on=["Capture photo code", "grid"], how="left")
    print(
        df[df.columns.difference(["grid"])]
        .isnull()
        .groupby(df.grid)
        .sum()
        .astype(int)["filepath"]
    )

    # Modify 'Capture photo code' using the marks and Capture # of those photos unable to be located
    df["Capture photo code"] = np.where(
        df["filepath"].isna(),
        df["Back left mark"].astype(str).apply(lambda x: "0" if "?" in x else x)
        + df["Back right mark"].astype(str).apply(lambda x: "0" if "?" in x else x)
        + df["Face left mark"].astype(str).apply(lambda x: "0" if "?" in x else x)
        + df["Face right mark"].astype(str).apply(lambda x: "0" if "?" in x else x)
        + "-"
        + df["Capture #"].astype(int).astype(str),
        df["Capture photo code"],
    )

    # Add filepath of the photos to each frog identification again with the updated 'Capture photo code'
    df = df.drop(
        columns=list(set(list(frog_df.columns)) - set(["Capture photo code", "grid"]))
    ).merge(frog_df, on=["Capture photo code", "grid"], how="left")
    print(
        df[df.columns.difference(["grid"])]
        .isnull()
        .groupby(df.grid)
        .sum()
        .astype(int)["filepath"]
    )

    # Modify 'Capture photo code' using the marks and Capture # of those photos unable to be located
    df["Capture photo code"] = np.where(
        df["filepath"].isna(),
        df["Back left mark"].astype(str).apply(lambda x: "1" if "?" in x else x)
        + df["Back right mark"].astype(str).apply(lambda x: "1" if "?" in x else x)
        + df["Face left mark"].astype(str).apply(lambda x: "1" if "?" in x else x)
        + df["Face right mark"].astype(str).apply(lambda x: "1" if "?" in x else x)
        + "-"
        + df["Capture #"].astype(int).astype(str),
        df["Capture photo code"],
    )

    # Add filepath of the photos to each frog identification again with the updated 'Capture photo code'
    df = df.drop(
        columns=list(set(list(frog_df.columns)) - set(["Capture photo code", "grid"]))
    ).merge(frog_df, on=["Capture photo code", "grid"], how="left")
    print(
        df[df.columns.difference(["grid"])]
        .isnull()
        .groupby(df.grid)
        .sum()
        .astype(int)["filepath"]
    )

    df = df.rename(columns={"Capture photo code": "updated Capture photo code"})

    df = df.drop_duplicates(["Capture #", "grid"])

    new_df = frog_id_df.merge(
        df[["Capture #", "grid", "updated Capture photo code"]],
        on=["Capture #", "grid"],
        how="left",
    )

    # make sure if empty original values are used
    new_df["updated Capture photo code"] = np.where(
        new_df["updated Capture photo code"].isna(),
        new_df["Capture photo code"],
        new_df["updated Capture photo code"],
    )

    new_df["updated Capture photo code"] = np.where(
        new_df["updated Capture photo code"].isna(),
        new_df["Capture photo code"],
        new_df["updated Capture photo code"],
    )

    new_df["different Capture photo Code"] = np.where(
        new_df["Capture photo code"] == new_df["updated Capture photo code"], 0, 1
    )

    # Closest match between the Capture photo code and filenames
    new_df[new_df["grid"] == "Grid A"].drop(columns=["grid"]).to_csv(
        "victor_reviewed_grid_a.csv"
    )
    new_df[new_df["grid"] == "Grid B"].drop(columns=["grid"]).to_csv(
        "victor_reviewed_grid_b.csv"
    )
    new_df[new_df["grid"] == "Grid C"].drop(columns=["grid"]).to_csv(
        "victor_reviewed_grid_c.csv"
    )
    new_df[new_df["grid"] == "Grid D"].drop(columns=["grid"]).to_csv(
        "victor_reviewed_grid_d.csv"
    )
    new_df[new_df["grid"] == "Pukeokahu Frog Monitoring"].drop(columns=["grid"]).to_csv(
        "victor_reviewed_pukeokahu.csv"
    )

    # Missing photos
    df[(df["grid"] == "Grid A") & (df["filepath"].isna())].to_csv("missing_grid_a.csv")
    df[(df["grid"] == "Grid B") & (df["filepath"].isna())].to_csv("missing_grid_b.csv")
    df[(df["grid"] == "Grid C") & (df["filepath"].isna())].to_csv("missing_grid_c.csv")
    df[(df["grid"] == "Grid D") & (df["filepath"].isna())].to_csv("missing_grid_d.csv")
    df[(df["grid"] == "Pukeokahu Frog Monitoring") & (df["filepath"].isna())].to_csv(
        "missing_pukeokahu.csv"
    )

    # Add filepath info for whareorino_df
    whareorino_df_a_complete_df = whareorino_df["Grid A"].merge(
        frog_df, on=["Capture photo code", "grid"], how="left"
    )

    whareorino_df_b_complete_df = whareorino_df["Grid B"].merge(
        frog_df, on=["Capture photo code", "grid"], how="left"
    )

    whareorino_df_c_complete_df = whareorino_df["Grid C"].merge(
        frog_df, on=["Capture photo code", "grid"], how="left"
    )

    whareorino_df_d_complete_df = whareorino_df["Grid D"].merge(
        frog_df, on=["Capture photo code", "grid"], how="left"
    )

    # Add filepath info for pukeokahu
    pukeokahu_complete_df = pukeokahu_df["MR Data"].merge(
        frog_df, on=["Capture photo code", "grid"], how="left"
    )


def check_column_consistency(
    pukeokahu_df: pd.DataFrame, whareorino_df: pd.DataFrame
) -> None:
    """Check for consistent column names"""

    # AB
    col_diff = list(
        set(whareorino_df["Grid A"].columns) - set(whareorino_df["Grid B"].columns)
    )
    if col_diff:
        print("Differences between A and B", col_diff)

    # BA
    col_diff = list(
        set(whareorino_df["Grid B"].columns) - set(whareorino_df["Grid A"].columns)
    )
    if col_diff:
        print("Differences between B and A", col_diff)

    # AC
    col_diff = list(
        set(whareorino_df["Grid A"].columns) - set(whareorino_df["Grid C"].columns)
    )
    if col_diff:
        print("Differences between A and C", col_diff)

    # CA
    col_diff = list(
        set(whareorino_df["Grid C"].columns) - set(whareorino_df["Grid A"].columns)
    )
    if col_diff:
        print("Differences between C and A", col_diff)

    # AD
    col_diff = list(
        set(whareorino_df["Grid A"].columns) - set(whareorino_df["Grid D"].columns)
    )
    if col_diff:
        print("Differences between A and D", col_diff)

    # DA
    col_diff = list(
        set(whareorino_df["Grid D"].columns) - set(whareorino_df["Grid A"].columns)
    )
    if col_diff:
        print("Differences between D and A", col_diff)

    # AP
    col_diff = list(
        set(whareorino_df["Grid A"].columns) - set(pukeokahu_df["MR Data"].columns)
    )
    if col_diff:
        print("Differences between A and pukeokahu", col_diff)

    # PA
    col_diff = list(
        set(pukeokahu_df["MR Data"].columns) - set(whareorino_df["Grid A"].columns)
    )
    if col_diff:
        print("Differences between pukeokahu and A", col_diff)
