import joblib
import re
import pandas as pd
from datetime import datetime
import unidecode
import numpy as np
import os
import shutil

def remove_letters(x):

    x = str(re.sub(r'[a-zA-Z]', '', str(x)))
    return x

def decimalize(x, decimal = 2):
    x = remove_letters(x)
    x = x.rstrip(" ").lstrip(" ")
    if len(x) == 4 or len(x) == 5:
        if "." or "," in x:
            x = x.replace(".","").replace(",","")
        x = round(int(x)/1000, decimal)
    return x

def detect_numeric(df):

    vars = []
    for col in df.columns:
        try:
            pd.to_numeric(df[col])
            vars.append(col)
        except:
            pass
    return vars

def remove_accents(df, variable):

    df[variable] = [unidecode(x.lower()).replace("nan", np.nan) for x in df[variable].astype(str)]

def convert_to_numeric(row, type):
    """
    df["año2"] = df["año"].apply(lambda x: convert_to_numeric(x, "int"))
    df["año2"].fillna(0).astype(int)
    """
    try:
        if type == "float":
            return float(row)
        elif type == "int":
            return int(row)
    except:
        return np.nan

def GetActualDate():
    actual_date = datetime.now()
    month = str(actual_date.month)
    month = np.where(len(month) == 1, "0" + month, month)
    day = str(actual_date.day)
    day = np.where(len(day) == 1, "0" + day, day)
    actual_date = str(actual_date.year) +  str(month) + str(day)
    return actual_date

def ClearFolder(folder):

    for archivo in os.listdir(folder):
        ruta_completa = os.path.join(folder, archivo)
        if os.path.isfile(ruta_completa):
            os.remove(ruta_completa)
        elif os.path.isdir(ruta_completa):
            shutil.rmtree(ruta_completa)

class pkl:
    @staticmethod
    def Save(obj, path):
        joblib.dump(obj, path)

    @staticmethod
    def Load(path):
        obj = joblib.load(path)
        return obj
    
    @staticmethod
    def Remove(path):
        import os
        os.remove(path)

def FixTildes(texto):
    import pandas as pd
    """
    df["tipo_de_carroceria"].apply(agregar_tildes)
    """
    sustituciones = {
        "√°": "á",
        "√©": "é",
        "√≥": "í",
        "√≤": "ó",
        "√º": "ú",
        "√Ñ": "ñ"}

    for simbolo, tilde in sustituciones.items():
        if pd.isnull(texto) == False:
            texto = texto.replace(simbolo, tilde)
    return texto


