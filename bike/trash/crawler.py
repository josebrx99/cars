import pandas as pd
import numpy as np
import requests
from lxml import html
from bs4 import BeautifulSoup
import requests
import time
from tqdm import tqdm
from datetime import datetime
from unidecode import unidecode
import warnings
import functions as func
warnings.filterwarnings("ignore")
pd.options.display.max_columns = None

headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) \nChrome/103.0.5060.114 Safari/537.36 Edg/103.0.1264.49"}


def CompleteAtributes(df):
    def _transmision(row):
        if pd.isnull(row["transmision"]):
            if "automatic" in str(row["version"]) or "automatic" in str(row["nombre"]):
                return "Automática"
            elif "mecanic" in str(row["version"]) or "mecanic" in str(row["nombre"]):
                return "Mecánica"
            else:
                return np.nan
        else:
            return row["transmision"]

    df["transmision"] = df.apply(_transmision, axis=1)

    def _carroceria(row):
        if pd.isnull(row["tipo_de_carroceria"]):
            if "sedan" in str(row["version"]) or "sedan" in str(row["nombre"]):
                return "Sedán"
            elif "coupe" in str(row["version"]) or "coupe" in str(row["nombre"]):
                return "Coupé"
            elif "hatchback" in str(row["version"]) or "hatchback" in str(row["nombre"]):
                return "Hatchback"
            elif "camioneta" in str(row["version"]) or "camioneta" in str(row["nombre"]):
                return "Camioneta"
            elif "cabriolet" in str(row["version"]) or "cabriolet" in str(row["nombre"]):
                return "Cabriolet"
            elif "wagon" in str(row["version"]) or "wagon" in str(row["nombre"]):
                return "Camioneta"
            elif "van" in str(row["version"]) or "van" in str(row["nombre"]):
                return "Van"
            elif "minivan" in str(row["version"]) or "minivan" in str(row["nombre"]):
                return "Minivan"
            elif "doble cabina" in str(row["version"]) or "doble cabina" in str(row["nombre"]):
                return "Pick-Up"
            elif "furgon" in str(row["version"]) or "furgon" in str(row["nombre"]):
                return "Furgón"
            elif "roadster" in str(row["version"]) or "roadster" in str(row["nombre"]):
                return "Roadster"
            elif "suv" in str(row["version"]) or "suv" in str(row["nombre"]):
                return "Camioneta"
            else:
                return np.nan
        else:
            return row["tipo_de_carroceria"]
    df["tipo_de_carroceria"] = df.apply(_carroceria, axis = 1)

    def _combustible(row):
        if pd.isnull(row["tipo_de_combustible"]):
            if "gasolina" in str(row["version"]) or "gasolina" in str(row["nombre"]):
                return "Gasolina"
            elif "diesel" in str(row["version"]) or "diesel" in str(row["nombre"]):
                return "Diésel"
            elif "hibrid" in str(row["version"]) or "hibrid" in str(row["nombre"]):
                return "Híbrido"
            elif "electric" in str(row["version"]) or "electric" in str(row["nombre"]):
                return "Eléctrico"
            else:
                return np.nan
        else:
            return row["tipo_de_combustible"]

    df["tipo_de_combustible"] = df.apply(_combustible, axis=1)

    df["modelo"] = df.apply(HomologModel, axis=1)

    df["marca"] = df.apply(HomologBrand, axis=1)
    return df

def HomologModel(row):
    years_list = [str(datetime.now().year - i) for i in range(50)]

    marca = str(row["marca"])
    modelo = str(row["modelo"]).lower()
    if modelo in years_list:
        return np.nan
    elif marca == "Audi":
        if "q5" in modelo:
            return "Q5"
        elif "q3" in modelo:
            return "Q3"
        else:
            return row["modelo"]
    elif marca == "Chevrolet":
        if "tracker" in modelo:
            return "Tracker"
        elif "blazer" in modelo:
            return "Blazer"
        elif "luv" in modelo:
            return "LUV"
        elif "aveo" in modelo:
            return "Aveo"
        elif "joy" in modelo:
            return "Joy"
        else:
            return row["modelo"]
    elif marca == "Renault":
        if "megane" in modelo or "mégane" in modelo:
            return "Megane"
        elif "stepway" in modelo:
            return "Stepway"
        elif "clio" in modelo:
            return "Clio"
        elif "koleos" in modelo:
            return "Koleos"
        else:
            return row["modelo"]
    elif marca == "Nissan":
        if "x trail" in modelo or "xtrail" in modelo:
            return "X-Trail"
        elif "tiida" in modelo:
            return "Tiida"
        elif "np300" in modelo or "frontier" in modelo:
            return "Frontier"
        else:
            return row["modelo"]
    elif marca == "Ford":
        if "mustang" in modelo:
            return "Mustang"
        elif "f-150" in modelo or "f150" in modelo:
            return "F-150"
        elif "bronco" in modelo:
            return "Bronco"
        elif "fiesta" in modelo:
            return "Fiesta"
        else:
            return row["modelo"]
    elif marca == "Mazda":
        if "Mazda 2" in marca + " " + modelo:
            return "2"
        elif "Mazda 3" in marca + " " + modelo:
            return "3"
        else:
            return row["modelo"]
    elif marca == "Mercedes-Benz":
        if "gle" in modelo:
            return "Clase GLE"
        elif "gl" in modelo:
            return "Clase GL"
        elif "glc" in modelo:
            return "Clase GLC"
        else:
            return row["modelo"]
    elif marca == "Toyota":
        if "rav4" in modelo:
            return "RAV4"
        elif "prado" in modelo:
            return "Prado"
        elif "sahara" in modelo:
            return "Sahara"
        elif "yaris cross" in modelo:
            return "Yaris Cross"
        elif "yaris" in modelo and "yaris cross" not in modelo:
            return "Yaris"
        elif "fortuner" in modelo:
            return "Fortuner"
        elif "corolla cross" in modelo:
            return "Corolla Cross"
        elif "hilux" in modelo:
            return "Hilux"
        else:
            return row["modelo"]
    elif marca == "Volkswagen":
        if "t-cross" in modelo:
            return "T-Cross"
        elif "escarabajo" in modelo:
            return "Escarabajo"
        else:
            return row["modelo"]
    elif marca == "BMW":
        if "ix3" in modelo:
            return "iX3"
        elif "120i" in modelo:
            return "120i"
        elif "218" in modelo:
            return "218"
        elif "120i" in modelo:
            return "120i"
        elif "ix3" in modelo:
            return "Ix3"
        elif "420i" in modelo:
            return "420i"
        else:
            return row["modelo"]
    elif marca == "Kia":
        if "rio" in modelo:
            return "Rio"
        elif "sportage" in modelo:
            return "Sportage"
        elif "picanto" in modelo:
            return "Picanto"
        elif "carens" in modelo:
            return "Carens"
        elif "cerato" in modelo:
            return "Cerato"
        elif "new sportage" in modelo:
            return "New Sportage"
        else:
            return row["modelo"]
    elif marca == "Suzuki":
        if "swift" in modelo:
            return "Swift"
        elif "vitara" in modelo:
            return "Vitara"
        elif "grand vitara" in modelo:
            return "Grand Vitara"
        elif "jimny" in modelo:
            return "Jimny"
        else:
            return row["modelo"]
    elif marca == "Hyundai":
        if "kona" in modelo:
            return "Kona"
        elif "tucson" in modelo:
            return "Tucson"
        elif "venue" in modelo:
            return "Venue"
        elif "accent" in modelo:
            return "Accent"
        elif "atos" in modelo:
            return "Atos"
        elif "i10" in modelo:
            return "i10"
        else:
            return row["modelo"]
    elif marca == "Honda":
        if "civic" in modelo:
            return "Civic"
        else:
            return row["modelo"]

    else:
        return row["modelo"]
    
def HomologBrand(row):
    marca = str(row["marca"])
    if "DFSK" in marca:
        return "DFSK"
    elif "Mercedes-Benz" in marca:
        return "Mercedes-Benz"
    elif "Nissan" in marca:
        return "Nissan"
    else:
        return row["marca"]

def GetCarAtributes(url, debug = "OFF"):
    """
    Función que obtiene los atributos de un vehículo
    """

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        #print(f'Error HTTP ocurrido: {e}, {url}')
        df_ofert_temp = pd.DataFrame({"estado":[0], "url":url})
        _error_server = 1
    except Exception as e:
        #print(f'Error inesperado ocurrido: {e}, {url}')
        df_ofert_temp = pd.DataFrame({"estado":[0], "url":url})
        _error_server = 1
    else:
        time1 = time.time()
        _error_server = 0
        tree = html.fromstring(response.content)

        def _obtain_xpath(tree, xpath):
            _object = tree.xpath(xpath)
            if _object != []: 
                _object = _object[0].text_content().strip() 
            else: _object = "missing"
            return  _object
        
        nombre = _obtain_xpath(tree, '//*[@id="header"]/div/div[2]/h1')
        precio = _obtain_xpath(tree,'//*[@id="price"]/div/div/div/span/span/span[2]')
        #precio = int(tree.xpath('//*[@id="price"]/div/div/div/span/span/span[2]')[0].text_content().strip().replace(".",""))
        precio = np.where(precio != "missing", precio.replace(".",""), 0)
        año_km_fecha = _obtain_xpath(tree, '//*[@id="header"]/div/div[1]/span')
        # año_km_fecha = tree.xpath('//*[@id="header"]/div/div[1]/span')[0].text_content().strip()
        año = np.where(precio != "missing", año_km_fecha.split("|")[0].replace(" ",""), 0)
        try:
            km = np.where(precio != "missing", año_km_fecha.split("|")[1].split("·")[0].replace(" ","").replace("km","").replace(".",""), 0)
        except:
            km = 0
        try:
            fecha = año_km_fecha.split("|")[1].split("·")[1].replace(" Publicado hace ","")
        except:
            fecha = "missing"
        id = _obtain_xpath(tree, '//*[@id="denounce"]/div/p/span')
        id = id.replace("#","")
        ubicacion = tree.xpath('//*[@id="seller_profile"]')
        if ubicacion != []:
            try:
                ubicacion = tree.xpath('//*[@id="seller_profile"]')[0].text_content().strip().replace("Información de la tienda","").split("Ubicación del vehículo")[1].split("Ver teléfono")[0]
                ubicacion = ubicacion.split("-")
            except:
                ubicacion = ["missing"]*3
        else:
            ubicacion = ["missing"]*3

        if len(ubicacion) == 2:
            barrio = np.nan
            ciudad = ubicacion[0]
            departamento = ubicacion[1]
        elif len(ubicacion) == 3:
            barrio = ubicacion[0].rstrip(" ")
            ciudad = ubicacion[1].rstrip(" ").lstrip(" ")
            departamento = ubicacion[2].lstrip(" ")
        elif len(ubicacion) == 4:
            barrio = ubicacion[1].rstrip(" ")
            ciudad = ubicacion[2].rstrip(" ").lstrip(" ")
            departamento = ubicacion[3].lstrip(" ")
        elif len(ubicacion) == 5:
            barrio = ubicacion[2].rstrip(" ")
            ciudad = ubicacion[3].rstrip(" ").lstrip(" ")
            departamento = ubicacion[4].lstrip(" ")
        else:
            print("ERROR: ubicación con más de 5 elementos", url)
            barrio, ciudad, departamento = np.nan
        barrio = np.where(barrio == "missing", np.nan, barrio)
        ciudad = np.where(ciudad == "missing", np.nan, ciudad)
        departamento = np.where(departamento == "missing", np.nan, departamento)

        if debug == "ON": print(f"atributos1: {time.time() - time1:.2f} seg")
        time1 = time.time()

        # Ingest class table-andes
        soup = BeautifulSoup(response.content, 'html.parser')
        andes_table = soup.find("table", class_="andes-table")

        def _ingest_table_andes(rows):
            _names_atributes = []
            for i in range(0, len(rows)):
                _atribute = [th.text.strip() for th in rows[i].find_all("th")]
                _names_atributes.append(_atribute[0])

            _atributes = []
            for i in rows: 
                i = i.find_all("td")
                i = [i.text.strip() for i in i]
                _atributes.append(i[0])

            _atributes = pd.DataFrame([_atributes], columns=_names_atributes)

            cols_del = ["Kilómetros", "Año"]
            _cols_to_delete = [col for col in cols_del if col in _atributes.columns]
            _atributes.drop(columns=_cols_to_delete, inplace=True)

            return _atributes

        if andes_table == None: # Try again ingest
            time.sleep(1)
            for k in range(0, 1):
                try:
                    response = requests.get(url, headers=headers)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    andes_table = soup.find("table", class_="andes-table")
                    rows_andes_table = andes_table.find_all("tr")
                    status = 1
                    atributes = _ingest_table_andes(rows_andes_table)
                except:
                    if debug == "ON": print("error andes-table")
                    atributes = pd.DataFrame()

                if "Marca" in atributes:
                    break
                #time.sleep(1.5)

            if andes_table == None:
                if debug == "ON": print("empty")
                status = 0
                atributes = pd.DataFrame()
        else:
            rows_andes_table = andes_table.find_all("tr")
            status = 1
            atributes = _ingest_table_andes(rows_andes_table)
        if debug == "ON": print(f"andes-table: {time.time() - time1:.2f} seg"); print("-"*30)
        time1 = time.time()

        dtm_today = datetime.now()
        dtm_today = dtm_today.strftime('%Y-%m-%d')

        df_ofert_temp = pd.DataFrame({"estado":status, "nombre":[nombre], "precio":[precio], "año":[año], "km":[km], "barrio":[barrio],
                                       "ciudad":ciudad, "departamento":departamento, "fecha":[fecha], "url":url, "dtm_etl": dtm_today, 
                                       "id_ofert": id})
        df_ofert_temp = pd.concat([df_ofert_temp, atributes], axis = 1)
        df_ofert_temp.columns = [x.lower() for x in df_ofert_temp.columns]
        df_ofert_temp.columns = [unidecode(x.lower().replace(" ","_")) for x in df_ofert_temp.columns]
        df_ofert_temp.rename(columns = {"ano":"año"}, inplace = True)
        df_ofert_temp = df_ofert_temp.replace("missing", np.nan)

    return df_ofert_temp, _error_server

def RefineAtributes(df, debug = "OFF"):
    """
    Función que refina y homologa los atributos del vehículo
    """
    time1 = time.time()
    def _refine_motor(x):
        x = func.remove_letters(x)
        x = x.rstrip(" ").lstrip(" ")
        if x.isdigit() or x.replace('.', '', 1).isdigit():
            pass
        else:
            x = "0"

        if (3 <= len(x) <= 4) and ("." not in x and "," not in x):
            x = round(float(x)/1000, 2)
        elif (len(x) >= 5) and ("." in x[4] or "," in x[4]):
            x = round(float(x)/1000, 2)
        elif (len(x) >= 5) and ("." in x[1] or "," in x[1]):
            x = round(float(x), 2)
        else:
            pass

        if float(x) <= 0.7 or float(x) >= 8:
            x = np.nan
        return x
    df["motor"] = [_refine_motor(x) for x in df["motor"]]
    df["motor"] = df["motor"].replace({"":np.nan, "0":np.nan})
    df["motor"] = pd.to_numeric(df["motor"], errors = "coerce")

    df["barrio"] = df["barrio"].fillna("missing").astype(str)
    df["ciudad"] = df["ciudad"].fillna("missing").astype(str)
    df["departamento"] = df["departamento"].fillna("missing").astype(str)

    df["barrio"] = [x.rstrip(" ").lstrip(" ") for x in df["barrio"]]
    df["barrio"] = df["barrio"].replace({"nan":np.nan})

    vars = ["ciudad", "departamento"]
    for col in vars:
        df[col] = [x.split("Ir a la tienda oficial")[0] for x in df[col]]
        df[col] = [x.split("Ir a la página")[0] for x in df[col]]
        df[col] = [x.rstrip(" ").lstrip(" ") for x in df[col].astype(str)]
        df[col] = df[col].replace({"nan":np.nan})
    df["ciudad"] = np.where(df["departamento"] == "Bogotá D.C.", "Bogotá D.C.", df["ciudad"])

    vars_null = ["tipo_de_carroceria", "tipo_de_combustible", "transmision", "color"]
    for col in vars_null:
        df[col] = df[col].fillna("missing")

    df["nombre"] = [unidecode(x.lower()) for x in df["nombre"].astype(str)]
    df["nombre"] = df["nombre"].replace({"nan":np.nan})

    df["version"] = [unidecode(x.lower()) for x in df["version"].astype(str)]
    df["version"] = df["version"].replace({"nan":np.nan})

    #df["con_cámara_de_reversa"] = np.where(df["con_cámara_de_reversa"] == "Sí", 1, np.where(df["con_cámara_de_reversa"] == "No", 0, np.nan))
    df["con_camara_de_reversa"] = np.where(df["con_camara_de_reversa"] == "Sí", 1, 0)

    dicc_carroceria = {"SUV":"Camioneta", "WAGON":"Camioneta", "Station Wagon":"Camioneta", 'Off-Road':"Camioneta", 
                       'Minivan':"Van"}
    df["tipo_de_carroceria"] = df["tipo_de_carroceria"].replace(dicc_carroceria)
    df["tipo_de_carroceria"] = ["Pick-Up" if "doble cabina" in x.lower() else x for x in df["tipo_de_carroceria"]]
    df["tipo_de_carroceria"] = np.where(~df["tipo_de_carroceria"].isin(['Camioneta', 'Sedán', 'Hatchback', 'Furgón', 'Pick-Up', 'Coupé',
                                                                       'Convertible', 'Van', 'Roadster']), np.nan, df["tipo_de_carroceria"])

    dicc_color = {"Manual":"Mecánica"}
    df["transmision"] = df["transmision"].replace(dicc_color)
    df["transmision"] = np.where(~df["transmision"].isin(['Automática', 'Mecánica']), np.nan, df["transmision"])

    df["tipo_de_combustible"] = np.where(~df["tipo_de_combustible"].isin(['Gasolina', 'Gasolina y gas', 'Diésel', 'Híbrido', 'Eléctrico']), 
                                         np.nan, df["transmision"])

    dicc_color = {"Gris":"Plateado", "Celeste":"Azul"}
    df["color"] = df["color"].replace(dicc_color)

    df["antiguedad"] = np.where(datetime.today().year - df["año"] == 0, 1, datetime.today().year - df["año"])

    df["puertas"] = df["puertas"].fillna(0).astype(int).astype(str)
    df["puertas"] = np.where(df["puertas"].isin(["2","3"]), "2_3", np.where(df["puertas"].isin(["4","5"]), "4_5", np.nan))
    df["puertas"] = df["puertas"].replace("nan", np.nan)

    df["km_por_año"] = round(df["km"] / df["antiguedad"], 2)

    df["precio"] = round(df["precio"]/1000000, 3)

    df["marca_modelo"] = df["marca"] + " " + df["modelo"]

    df["sk_id"] = [i for i in range(0, df.shape[0])]

    df["año"] = df["año"].fillna(999).astype(int).replace(999,np.nan)
    df["km"] = df["km"].fillna(99999999).astype(int).replace(99999999,np.nan)
    df["antiguedad"] = df["antiguedad"].fillna(999).astype(int).replace(999,np.nan)
    df["ultimo_digito_de_la_placa"] = df["ultimo_digito_de_la_placa"].fillna(999).astype(int).replace(999,np.nan)

    df = CompleteAtributes(df)
    df = df.replace("nan", np.nan).replace("missing", np.nan)

    if debug == "ON": print(f"RefineAtributes: {time.time() - time1:.2f} seg")
    time1 = time.time()

    return df


# def Main():
