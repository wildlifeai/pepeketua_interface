import argparse
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import sqlalchemy as db
from dateutil.parser import parse


def is_date(obj: Any) -> bool:
    """
    Return whether an object can be interpreted as a date.
    :param obj: object suspected of being a date
    """
    if (obj is np.nan) or (obj is None):
        return False
    else:
        if isinstance(obj, datetime):
            return True
        try:
            parse(obj, fuzzy=True)
            return True
        except ValueError:
            return False


def is_number(obj: Any) -> bool:
    """
    Return whether an object is a valid int or float
    :param obj:
    :return:
    """
    if (obj is np.nan) or (obj is np.inf) or (obj is -np.inf):
        return False
    elif isinstance(obj, int):
        return True
    elif isinstance(obj, float):
        return True
    else:
        return False


def filter_excel_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter capture sheets from empty rows by selecting only rows with valid "Date" and "Capture #" entries
    :param df: DataFrame containing data from capture excel sheets
    :return: Return filtered df
    """
    capture_idx = df.loc[:, "Capture #"].apply(is_number)
    date_idx = df.loc[:, "Date"].apply(is_date)
    idx = capture_idx & date_idx
    return df[idx].reset_index(drop=True)


def dump_df_to_postgres(df: pd.DataFrame, if_exists: str = "replace") -> Optional[int]:
    engine = db.create_engine("postgresql://lioruzan:nyudEce5@localhost/frogs")
    with engine.connect() as con:
        ret = df.to_sql("frogs", con, if_exists=if_exists, index=False)
    return ret


def main(whareorino_excel_file: str, pukeokahu_excel_file: str) -> None:
    # build dataframe db, including Grid column
    dfs = []

    # parse whareorino
    grids = ["Grid A", "Grid B", "Grid C", "Grid D"]
    for grid in grids:
        df = pd.read_excel(whareorino_excel_file, sheet_name=grid)
        df = filter_excel_sheet(df)
        df["Grid"] = grid
        dfs.append(df)

    # parse pukeokahu
    df = pd.read_excel(pukeokahu_excel_file, sheet_name="MR Data")
    df = filter_excel_sheet(df)
    df["Grid"] = "Pukeokahu Frog Monitoring"
    dfs.append(df)

    full_data = pd.concat(dfs).reset_index(drop=True)

    # sort out image filename column <-> image correspondants
    pass

    # dump to sql
    dump_df_to_postgres(full_data)

    # save images to lmdb
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("whareorino_excel_file")
    parser.add_argument("pukeokahu_excel_file")
    args = parser.parse_args()
    main(args.whareorino_excel, args.pukeokahu_excel)
