import os
import logging
import chardet
import pandas as pd
from pandas import DataFrame
from typing import List, Dict

logger = logging.getLogger(__name__)

'''
Leemos nuestro Data_dict
'''

user = os.environ.get('USERNAME').lower()
local_path = f'C:\\Users\\{user}\\Desktop\\workplace\\prce-mkt-mtr-Falsos-Reemplazos'
file_path = os.path.join(local_path, "conf", r"falsos_reemplazos_dict.csv")

with open(file_path, "rb") as f:
    result = chardet.detect(f.read())
df_dict = pd.read_csv(
    file_path,
    dtype = str,
    encoding = result["encoding"],
    sep = ";",
    keep_default_na = False,)

def load(df, filter_cols=False) -> pd.DataFrame:
    """
    This method loads the data dictionary as a DataFrame, optionally filtered by the columns to import for
    a specific type.
    :param filter_cols: Flag to filter for the columns marked as import for the model, defaults to `False`
    :type filter_cols: bool
    :return: DataFrame as a data dictionary
    """
    if filter_cols:
        return df_dict[df_dict['IMPORT_CSV'] == "Y"]
    else:

        return df_dict

def get_import_columns(df) -> List[str]:

    df = load(df, filter_cols=True)
    import_cols = df["INPUT_FIELD"].values.tolist()
    print(len(import_cols))

    return import_cols

def get_rename_dict(df) -> Dict:
    """
    Method to get the rules to rename the input fields of the raw files as column names as specified in the
    data dictionary file

    :return: Dictionary where the key is the input field and the value is the column name of the dataframe
    """
    df = load(df, filter_cols=True)
    #df = df.load(filter_cols=True)
    rename_dict = dict(zip(df["INPUT_FIELD"], df["FEATURE_NAME"]))

    return rename_dict

def get_features_model(df, import_model: str="IMPORT_CSV") -> pd.DataFrame:

    df = df_dict[df_dict[import_model] == "Y"]
    df = df[df["FEATURE_TYPE"].isin(["Cat", "Num", "ID", "Date"])]
    import_cols = df["FEATURE_NAME"].values.tolist()

    return import_cols

def get_feature_type(df):

    features_dict = dict(zip(df_dict["FEATURE_NAME"], df_dict["FEATURE_TYPE"]))

    return features_dict

def get_dtypes_dict(df) -> Dict:
    #df = df_dict.load(filter_cols=True)
    df = load(df, filter_cols=True)
    df = df[df["OUT_DTYPE"] != "Date"]
    dtypes_dict = dict(zip(df["INPUT_FIELD"], df["OUT_DTYPE"]))

    return dtypes_dict

def get_features(df, FEATURE_TYPE: str = "All") -> List[str]:
    feature_types = ["All", "ID", "Num", "Cat"]
    if FEATURE_TYPE not in feature_types:
        raise ValueError("Argument not in types " + str(feature_types))
    # df = df_dict.load()
    df = load(df_dict, filter_cols=True)
    features = df.loc[
        df["FEATURE_TYPE"].isin(["Num", "Cat"]), "FEATURE_NAME"
    ].values.tolist()
    if FEATURE_TYPE != "All":
        features = df.loc[
            df["FEATURE_TYPE"] == FEATURE_TYPE, "FEATURE_NAME"
        ].values.tolist()

    return features

def date_format(df, feature_types, date_formats=None):
    if date_formats is None:
        date_formats = {}  # Default to an empty dictionary if date_formats is not provided

    for col, col_type in feature_types.items():
        if col_type == 'Date' and col in df.columns:
            if col in date_formats:
                try:
                    df[col] = pd.to_datetime(df[col], format=date_formats[col])
                except (TypeError, ValueError):
                    df[col] = pd.to_datetime(df[col], errors='coerce')  # Use default date parsing with 'coerce' option if format is not valid
            else:
                df[col] = pd.to_datetime(df[col], errors='coerce')  # Use default date parsing with 'coerce' option if format is not specified
    return df

def formato(df):

    df['FANUL'] = pd.to_datetime(df['FANUL'], errors='coerce')
    df['FIANUL'] = pd.to_datetime(df['FIANUL'], errors='coerce')
    df['TXCANAL'] = df['TXCANAL'].astype(object)
    df['PRIMAN'] = df['PRIMAN'].str.replace(".", "").str.replace(",", ".")
    df['prima_reemplazante'] = df['prima_reemplazante'].str.replace(".", "").str.replace(",", ".")
    df['prperiod'] = df['prperiod'].str.replace(".", "").str.replace(",", ".")
    df['SINOCU'] = df['SINOCU'].str.replace(".", "").str.replace(",", ".")
    df['POLIZAN'] = df['POLIZAN'].fillna(0).astype(int)
    df['APLICAN'] = df['APLICAN'].fillna(0).astype(int)
    df['CODPOST'] = df['CODPOST'].fillna(0).astype(int)
    df['CODMEDP'] = df['CODMEDP'].astype(str).str.rstrip('.0')
    df['GARANTIA_1'] = df['GARANTIA_1'].fillna(0).astype(int)
    df['GARANTIA_2'] = df['GARANTIA_2'].fillna(0).astype(int)
    df['GARANTIA_3'] = df['GARANTIA_3'].fillna(0).astype(int)
    df['GARANTIA_4'] = df['GARANTIA_4'].fillna(0).astype(int)
    df['GARANTIA_5'] = df['GARANTIA_5'].fillna(0).astype(int)
    df['GARANTIA_6'] = df['GARANTIA_6'].fillna(0).astype(int)
    df['GARANTIA_7'] = df['GARANTIA_7'].fillna(0).astype(int)
    df['GARANTIA_8'] = df['GARANTIA_8'].fillna(0).astype(int)
    df['Provincia'] = df['Provincia'].astype(str).str.rstrip('.0')
    df['Provincia'] = df['Provincia'].replace("**", "999 - NaN")
    df['Provincia'] = df['Provincia'].replace("", "999 - NaN")
    df['Provincia'] = df['Provincia'].replace("nan", "999 - NaN")
    df['sdad_total3p'] = df['sdad_total3p'].replace("0.00", "0.0")
    df['sdad_total_tom'] = df['sdad_total_tom'].replace("0.00", "0.0")
    df['sdad_total3p'] = df['sdad_total3p'].replace("nan", "0.0")
    df['sdad_total_tom'] = df['sdad_total_tom'].replace("nan", "0.0")
    df['sdad_total_tom'] = pd.to_numeric(df['sdad_total_tom'], errors='coerce')
    df['sdad_total3p'] = pd.to_numeric(df['sdad_total3p'], errors='coerce')
    df = df[~df['RAMO'].isin([1255, 1256])]
    df = df[df['prima_reemplazante'].isin(['0.00'])]
    df['ANTIG_VEHICULO'] = df['ANTIG_VEHICULO'].astype(str).str.rstrip('.0')
    df['ANTIG_VEHICULO'] = df['ANTIG_VEHICULO'].replace("nan", "0")
    df['ANTIG_VEHICULO'] = df['ANTIG_VEHICULO'].replace("", "0")
    df['priman_new_fr'] = df['priman_new_fr'].str.replace(".", "").str.replace(",", ".")
    df['APR_new'] = df['APR_new'].str.replace(".", "").str.replace(",", ".")
    df['TPR_new'] = df['TPR_new'].str.replace(".", "").str.replace(",", ".")
    df['APR_old'] = df['APR_old'].str.replace(".", "").str.replace(",", ".")
    df['TPR_old'] = df['TPR_old'].str.replace(".", "").str.replace(",", ".")
    df['Provincia'] = df['Provincia'].apply(lambda x: str(x).zfill(2))

    return df


def ratios(df):

    df['Ratio_Solicitud_Med'] = df['solicitud_Med'] / df['version_coti_Med']
    df['Ratio_Cotizacion_Med'] = df['coti_tec_Med'] / df['version_coti_Med']
    df['Ratio_Reemplazo_Med'] = df['Reemplazo_Med'] / df['poli_Med']
    df['Ratio_BM_Med'] = df['BM_ACTUAL_pol_Med']
    df['Ratio_Tirea_No_deseado_Med'] = df['tirea_NO_deseado_Med'] / df['poli_Med']
    df['Ratio_Diferente_CP_Med'] = df['diferente_CP_Med'] / df['poli_Med']
    df['Ratio_Poli_CAP_Med'] = df['Dcto_poli_CAP_Med'] / df['poli_Med']
    df['Ratio_Poli_VC_Med'] = df['Dcto_poli_VC_Med'] / df['poli_Med']
    df['Ratio_Poli_TRA_Med'] = df['Dcto_poli_TRA_Med'] / df['poli_Med']
    df['Ratio_Canc_NB_Med'] = df['Canc_NB_Med'] / df['poli_Med']
    df['Ratio_Canc_RW_Med'] = df['Canc_RW_Med'] / df['poli_Med']
    df['Ratio_Canc_impag_NB_Med'] = df['Canc_Impag_NB_Med'] / df['poli_Med']
    df['Ratio_Canc_impag_RW_Med'] = df['Canc_Impag_RW_Med'] / df['poli_Med']
    df['Ratio_Coste_Defensa_Med'] = df['Coste_defensa_Med'] / df['poli_Med']
    df['Ratio_Fracc_Med'] = df['FP_NO_Anual_Med'] / df['poli_Med']
    df['Ratio_Desbloqueo_Med'] = df['Desbloqueo_Med'] / df['poli_Med']
    df['Ratio_Desbl_FalsoReemp_Med'] = df['Desbl_FalsoReemp_Med'] / df['poli_Med']
    df['Ratio_Desbl_Menores_Med'] = df['Desbloqueo_Med'] / df['poli_Med']
    df['Ratio_Solicitud_gr_Med'] = df['solicitud_gr_Med'] / df['version_coti_gr_Med']
    df['Ratio_Cotizacion_gr_Med'] = df['coti_tec_gr_Med'] / df['version_coti_gr_Med']
    df['Ratio_Reemplazo_gr_Med'] = df['Reemplazo_gr_Med'] / df['poli_gr_Med']
    df['Ratio_BM_gr_Med'] = df['BM_ACTUAL_pol_gr_Med']
    df['Ratio_Tirea_No_deseado_gr_Med'] = df['tirea_NO_deseado_gr_Med'] / df['poli_gr_Med']
    df['Ratio_Diferente_CP_gr_Med'] = df['diferente_CP_gr_Med'] / df['poli_gr_Med']
    df['Ratio_Poli_CAP_gr_Med'] = df['Dcto_poli_CAP_gr_Med'] / df['poli_gr_Med']
    df['Ratio_Poli_VC_gr_Med'] = df['Dcto_poli_VC_gr_Med'] / df['poli_gr_Med']
    df['Ratio_Poli_TRA_gr_Med'] = df['Dcto_poli_TRA_gr_Med'] / df['poli_gr_Med']
    df['Ratio_Canc_NB_gr_Med'] = df['Canc_NB_gr_Med'] / df['poli_gr_Med']
    df['Ratio_Canc_RW_gr_Med'] = df['Canc_RW_gr_Med'] / df['poli_gr_Med']
    df['Ratio_Canc_impag_NB_gr_Med'] = df['Canc_Impag_NB_gr_Med'] / df['poli_gr_Med']
    df['Ratio_Canc_impag_RW_gr_Med'] = df['Canc_Impag_RW_gr_Med'] / df['poli_gr_Med']
    df['Ratio_Defensa_gr_Med'] = df['DEFENDIDA_gr_Med'] / df['DEFENDIBLE_gr_Med']
    df['Ratio_Coste_Defensa_gr_Med'] = df['Coste_defensa_gr_Med'] / df['poli_gr_Med']
    df['Ratio_Fracc_gr_Med'] = df['FP_NO_Anual_gr_Med'] / df['poli_gr_Med']
    df['Ratio_Desbloqueo_gr_Med'] = df['Desbloqueo_gr_Med'] / df['poli_gr_Med']
    df['Ratio_Desbl_FalsoReemp_gr_Med'] = df['Desbl_FalsoReemp_gr_Med'] / df['poli_gr_Med']
    df['Ratio_Desbl_Menores_gr_Med'] = df['Desbloqueo_gr_Med'] / df['poli_gr_Med']
    df['desv_Solicitud_Med'] = df['Ratio_Solicitud_Med'] / df['Ratio_Solicitud_gr_Med']
    df['desv_Cotizacion_Med'] = df['Ratio_Cotizacion_Med'] / df['Ratio_Cotizacion_gr_Med']
    df['desv_Reemplazo_Med'] = df['Ratio_Reemplazo_Med'] / df['Ratio_Reemplazo_gr_Med']
    df['desv_BM_Med'] = df['Ratio_BM_Med'] / df['Ratio_BM_gr_Med']
    df['desv_Tirea_No_deseado_Med'] = df['Ratio_Tirea_No_deseado_Med'] / df['Ratio_Diferente_CP_gr_Med']
    df['desv_Diferente_CP_Med'] = df['Ratio_Diferente_CP_Med'] / df['poli_gr_Med']
    df['desv_Poli_CAP_Med'] = df['Ratio_Poli_CAP_Med'] / df['Ratio_Poli_CAP_gr_Med']
    df['desv_Poli_VC_Med'] = df['Ratio_Poli_VC_Med'] / df['Ratio_Poli_VC_gr_Med']
    df['desv_Poli_TRA_Med'] = df['Ratio_Poli_TRA_Med'] / df['Ratio_Poli_TRA_gr_Med']
    df['desv_Canc_NB_Med'] = df['Ratio_Canc_NB_Med'] / df['Ratio_Canc_NB_gr_Med']
    df['desv_Canc_RW_Med'] = df['Ratio_Canc_RW_Med'] / df['Ratio_Canc_RW_gr_Med']
    df['desv_Canc_impag_NB_Med'] = df['Ratio_Canc_impag_NB_Med'] / df['Ratio_Canc_impag_NB_gr_Med']
    df['desv_Canc_impag_RW_Med'] = df['Ratio_Canc_impag_RW_Med'] / df['Ratio_Canc_impag_RW_gr_Med']
    df['desv_Defensa_Med'] = df['Ratio_DEFENSA_Med'] / df['Ratio_Defensa_gr_Med']
    df['desv_Coste_Defensa_Med'] = df['Ratio_Coste_Defensa_Med'] / df['Ratio_Coste_Defensa_gr_Med']
    df['desv_Fracc_Med'] = df['Ratio_Fracc_Med'] / df['Ratio_Fracc_gr_Med']
    df['desv_Desbloqueo_Med'] = df['Ratio_Desbloqueo_Med'] / df['Ratio_Desbloqueo_gr_Med']
    df['desv_Desbl_FalsoReemp_Med'] = df['Ratio_Desbl_FalsoReemp_Med'] / df['Ratio_Desbl_FalsoReemp_gr_Med']
    df['desv_Desbl_Menores_Med'] = df['Ratio_Desbl_Menores_Med'] / df['Ratio_Desbl_Menores_gr_Med']

    return df

class DataDictionary:
    """
    Class to manage the columns of the raw files by identifying the formats, the feature types and the target
    columns for each file. This class helps to maintain the data lineage from the data warehouse to the
    relational DB in the model
    """

    def rounding(df):

        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].round(2)

        return df

    def get_date_columns(self, output_name=True) -> List[str]:
        """
        Method to get the names of columns in the database that represent dates

        :param output_name: Flag to select if the of the columns should be the raw input name (input_field)
         or the loaded column in the train dataframe (feature_name), defaults to `True`
        :type output_name: bool
        :return: List of column names
        """
        df = self.load(filter_cols=True)
        if output_name:
            column_selected = "input_field"
        else:
            column_selected = "feature_name"
        date_columns = df.loc[
            df["out_dtype"] == "Date", column_selected
        ].values.tolist()
        return date_columns
