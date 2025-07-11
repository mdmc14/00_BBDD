import logging

import numpy as np
import pandas as pd

from pandas import DataFrame
import datetime

from src.data_dict import DataDictionary

logger = logging.getLogger(__name__)

def get_renames(dataset: pd.DataFrame, data_dict: DataDictionary) -> pd.DataFrame:
    """This function renames the columns of a given dataset based on the data dictionary

    :param dataset: The input dataframe to be renamed
    :type dataset: DataFrame
    :param data_dict: The data dictionary class containing the data catalogue and the renaming dictionary
    :type data_dict: DataDictionary
    :return: The dataframe with the renamed columns
    :rtype: DataFrame
    """

    return dataset.rename(columns=data_dict.get_rename_dict())


def parse_date(x: str) -> datetime:
    """This function parses a given date (in string format) and converts it to a datetime.datetime object

    :param x: The date in a string format
    :type x: str
    :return: The date in a datetime object
    :rtype: datetime
    """
    try:
        return datetime.datetime.strptime(x, "%d/%m/%Y")
    except TypeError:
        return np.nan

def process_data_dict(
    df_dict: DataFrame, close_date: datetime, hash_data_dict: str
) -> DataFrame:
    """This function reads the data dictionary file and creates a DataFrame with the format adapted to be
    uploaded to the data_dict table in the database

    :param df_dict: The path where the data dictionary is located
    :type df_dict: str
    :param close_date: The close date
    :type close_date: datetime
    :param hash_data_dict: The hash of the data dict file based on the file metadata
     that controls the last time modified
    :type hash_data_dict: str
    :return: The data dictionary stored in a DataFrame ready to be uploaded to the DB
    :rtype: DataFrame
    """

    columns_list = [
        "close_date",
        "process_date",
        "data_dict_hash",
        "input_field",
        "feature_name",
        "data_type",
        "out_dtype",
        "feature_type",
        "import_csv",
        "import_model",
        "dynamic_value",
        "explain_shallow_tree",
        "short_description",
        "long_description",
        "source_type",
        "source_table",
    ]

    df_dict.columns = [col.lower() for col in df_dict.columns]
    df_dict["close_date"] = close_date
    df_dict["process_date"] = datetime.datetime.now()
    df_dict["data_dict_hash"] = hash_data_dict

    return df_dict[[*columns_list]]
