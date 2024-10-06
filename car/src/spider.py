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
import random

import os
import sys
import homologations as ho

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')).replace("\\", "/")[:-4])
import models
import functions as func
import paths as ph

warnings.filterwarnings("ignore")
pd.options.display.max_columns = None

headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36 Edg/103.0.1264.49"}


actual_date = func.GetActualDate()
actual_year = datetime.today().year

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
    
    map_models = {
    "allego": "Allegro", "action": "Actyon", "actyon": "Actyon", "c4": "C4", "captur":"Captur", "CLA":"CLA",
    "CLA200":"CLA", "cooper":"Cooper", "Clase C":"Clase C", 'Clase A200':"Clase A200", "Clase CLA":"Clase CLA",
    'clase Eq':"Clase EQ", 'countryman':"Countryman", 'discovery se':"Discovery",'discovery sport':"Discovery",
    'ekotaxi':"Ekotaxi", 'f-pace':"F-Pace", 'glory 580':"Glory",'graviti':"Gravity", 'jetta':"Jetta",'joyear':"Joyear",
    'm 4':"M4", 'montero':"Montero", 'onix':"Onix",'outlander':"Outlander",'evoque':"Evoque",'velar':"Velar",
    'ranger':"Ranger",'rexton':"Rexton",'sonic':"Sonic",'song plus':"Song Plus",'tiguan':"Tiguan",'voleex':"Voleex",
    'wrangler unlimited':"Wrangler", 'x-trail':"X-Trail", 'xc40':"XC40",'xc90':"XC40", 'xpander':"Xpander",
    'xsara':"Xsara",'cx-50':"CX-50",'cx50':"CX-50",'cx 50':"CX-50",'montero':"Montero", 'quin':'Qin',
    "nice":"Nice", "fj":"FJ", 'A45 S':"A45S"}
    for key, value in map_models.items():
        if key in modelo:
            return value
    
    if marca == "Audi":
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
    elif "Mercedes-Benz" in marca or "Mercedes Benz" in marca:
        return "Mercedes-Benz"
    elif "Nissan" in marca:
        return "Nissan"
    elif "SEAT" in marca:
        return "Seat"
    else:
        return row["marca"]

def GetCarAtributes(url, debug = "OFF"):
    """
    Función que obtiene los atributos de un vehículo
    """

    try:
        proxy_list = [
           "138.121.142.154:8080"
        ]
        import random
        proxy = random.choice(proxy_list)
        proxy = {"http":"http://" + proxy, "https": "https://" + proxy}
        response = requests.get(url) #, proxies=proxy
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
        precio = 0 if pd.isnull(precio) else precio.replace(".","")

        año_km_fecha = _obtain_xpath(tree, '//*[@id="header"]/div/div[1]/span')
        # año_km_fecha = tree.xpath('//*[@id="header"]/div/div[1]/span')[0].text_content().strip()

        año_km_fecha = str(año_km_fecha)
        año = año_km_fecha.split("|")[0].replace(" ","")
        año = año.replace("missing","0")
        # try:
        #     km = str(km)
        #     km = np.where(str(precio) != "missing", int(año_km_fecha.split("|")[1].split("·")[0].replace(" ","").replace("km","").replace(".","")), 0)
        # except:
        #     km = 0

        try:
            fecha = año_km_fecha.split("|")[1].split("·")[1].replace(" Publicado hace ","")
        except:
            fecha = "missing"
        id = _obtain_xpath(tree, '//*[@id="denounce"]/div/p/span')
        id = str(id)
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
        # barrio = np.where(barrio == "missing", np.nan, barrio)
        # ciudad = np.where(ciudad == "missing", np.nan, ciudad)
        # departamento = np.where(departamento == "missing", np.nan, departamento)


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

            if "Kilómetros" in _atributes.columns:
                _atributes["km"] = _atributes["Kilómetros"]
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

        df_ofert_temp = pd.DataFrame({"estado":status, "nombre":[nombre], "precio":[precio], "año":[año],"barrio":[barrio],  # "km":[km], 
                                       "ciudad":ciudad, "departamento":departamento, "fecha":[fecha], "url":url, "dtm_etl": dtm_today, 
                                       "id_ofert": id})
        df_ofert_temp = pd.concat([df_ofert_temp, atributes], axis = 1)
        df_ofert_temp.columns = [x.lower() for x in df_ofert_temp.columns]
        df_ofert_temp.columns = [unidecode(x.lower().replace(" ","_")) for x in df_ofert_temp.columns]
        df_ofert_temp.rename(columns = {"ano":"año"}, inplace = True)
        #df_ofert_temp = df_ofert_temp.replace("missing", np.nan)
        
    #print(proxy)
    #print(df_ofert_temp)

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
                                         np.nan, df["tipo_de_combustible"])

    dicc_color = {"Gris":"Plateado", "Celeste":"Azul", 'Marrón oscuro':"Marrón", 'Azul oscuro':"Azul"}
    df["color"] = df["color"].replace(dicc_color)
    
    df["año"] = df["año"].apply(lambda x: func.convert_to_numeric(x, "int"))
    df["año"] = df["año"].fillna(actual_year).astype(int)
    
    df["antiguedad"] = np.where(actual_year - df["año"] == 0, 1, actual_year - df["año"])

    df["puertas"] = df["puertas"].fillna(0).astype(int).astype(str)
    df["puertas"] = np.where(df["puertas"].isin(["2","3"]), "2_3", "4_5")

    df["precio"] = round(df["precio"]/1000000, 3)

    df["marca_modelo"] = df["marca"] + " " + df["modelo"]

    df["km_por_año"] = round(df["km"].fillna(0).astype(float) / df["antiguedad"], 2)
    #df["sk_id"] = [i for i in range(0, df.shape[0])]

    df["año"] = df["año"].fillna(actual_year).astype(int)
    df["km"] = df["km"].fillna(0).astype(float).replace(0,np.nan)
    df["antiguedad"] = df["antiguedad"].fillna(0).astype(int).replace(0,np.nan)
    #df["ultimo_digito_de_la_placa"] = df["ultimo_digito_de_la_placa"].replace(None,999).fillna(999).astype(float).replace(999,np.nan)


    df = CompleteAtributes(df)
    df = df.replace("nan", np.nan).replace("missing", np.nan)


    # Filtros varios
    df = df[df["año"] != 0]
    df = df[df["km"] <= 600000]

    if debug == "ON": print(f"RefineAtributes: {time.time() - time1:.2f} seg")
    time1 = time.time()

    return df


def ObtainAllUrls(delay = 1):

    # Final filtrada
    marcas = sorted(['subaru','mazda','mini','dodge','citroen','mitsubishi','hyundai','ssangyong','mercedes-benz',
                     'nissan','bmw','chevrolet','toyota','jeep','renault','audi','chery','zotye','seat','skoda',
                     'opel','changan','volkswagen','fiat','ram','suzuki','honda','great-wall','kia','jac','volvo',
                     'jaguar','peugeot','land-rover','daihatsu','byd','ford'])
    #marcas = sorted(['subaru','mazda','citroen'])
    dim_ubicacion = ho.cities
    #dim_ubicacion = dim_ubicacion.sample()
    #dim_ubicacion = dim_ubicacion[dim_ubicacion["ciudad"] == "cali"]
    #dim_ubicacion = dim_ubicacion[dim_ubicacion["departamento"].isin(["valle-del-cauca", "bogota-dc", "antioquia"])]
    oferts_full = []

    departamentos = dim_ubicacion["departamento"].unique().tolist()
    for departamento in tqdm(departamentos, desc="Parseando ofertas por paginación"):
        #print(departamento)

        ciudades = dim_ubicacion[dim_ubicacion["departamento"] == departamento]["ciudad"].unique().tolist()
        for ciudad in ciudades:
            #print(ciudad)
            
            for marca in marcas:
                size = 0 # validacion
                for pag in range(49, 1969, 48):  # 48*n | El máximo de paginación por categoría es de 42, 1969 vehículos
                    url = f"https://carros.mercadolibre.com.co/{marca}/{departamento}/{ciudad}/_Desde_{pag}_ITEM*CONDITION_2230581_NoIndex_True"
                    #print(url)
                    response = requests.get(url, headers=headers)
                    content = response.content.decode('utf-8')
                    href_ofert = '//*[@id="root-app"]/div/div[3]/section/ol/li[1]/div'
                    #href_ofert = '//a[contains(@class, "poly-component__title")]'
                    
                    if "Escribe en el buscador lo que quieres encontrar." not in content and \
                        'No hay publicaciones que coincidan con tu búsqueda.' not in content:
                        tree = html.fromstring(content)
                        #urls_ofert_temp = [x.get('href') for x in tree.xpath(href_ofert)]
                        _urls_pag = []
                        elements = tree.xpath('//*[@id="root-app"]/div/div[3]/section/ol')
                        for element in elements:
                            items = element.xpath('.//a[@href]')
                            for item in items:
                                _urls_pag.append(item.get("href"))
                        oferts_full.append(_urls_pag)
                        time.sleep(delay)
                        size += len(_urls_pag)
                        #print(url)
                        #print("correcto")
                        #print(len(_urls_pag))
                    else:
                        base_url = "/".join(url.split("/")[:6])
                        response = requests.get(base_url, headers=headers)
                        tree = html.fromstring(response.content)
                        #urls_ofert_temp = [x.get('href') for x in tree.xpath(href_ofert)]
                        _urls_pag = []
                        elements = tree.xpath('//*[@id="root-app"]/div/div[3]/section/ol')
                        for element in elements:
                            items = element.xpath('.//a[@href]')
                            for item in items:
                                _urls_pag.append(item.get("href"))
                        oferts_full.append(_urls_pag)
                        time.sleep(delay)
                        size += len(_urls_pag)
                        #print(base_url)
                        #print("unica pagina")
                        #print(len(_urls_pag))
                        break
        time.sleep(10)

    oferts_full = [i for sublist in oferts_full for i in sublist]
    oferts_full = list(set(oferts_full))
    pd.DataFrame({"url":oferts_full}).to_parquet(f"{ph.data_car}/urls.parquet", compression = "gzip")
    print("Ofertas obtenidas:", len(oferts_full))

    return oferts_full


def ReadUrlsParquet():
    oferts_full = pd.read_parquet(f"{ph.data_car}/urls.parquet", engine='pyarrow')["url"].tolist()
    oferts_full = [i for sublist in oferts_full for i in sublist]
    oferts_full = list(set(oferts_full))
    print(len(oferts_full))


def ScrapUrl(oferts_full, 
             delay = 1,
             counter_file = 200,
             repetead = False
             ):
    df = pd.DataFrame()
    error_server = 0
    counter = 1
    change_time = False
    from datetime import datetime
    time1 = datetime.now()
    record_count = 0
    repetead = "_R" if repetead == True else "C" # C: registros correctos, R: registros que se volvieron a ingestar debido a error en andes-table

    for oferta_temp in tqdm(list(oferts_full), desc="Obteniendo variables"):
        try:
            df_oferta_temp, error_server_temp = GetCarAtributes(oferta_temp, debug = "OFF")
        except:
            df_oferta_temp = pd.DataFrame(columns = ["url", "estado"])
            error_server_temp = False
        df_oferta_temp = df_oferta_temp[df_oferta_temp["estado"] == 1]
        df = pd.concat([df, df_oferta_temp], axis = 0)

        # if "marca" in df_oferta_temp.columns:
        #     print("correcto")
        # else:
        #     print("incorrecto")

        error_server += error_server_temp
        if error_server > len(oferts_full)*0.02:
            save_model = False
            raise ValueError("❌ ERROR: Error Server > 2% registers:", error_server)
            break

        #if "marca" in df_oferta_temp:

        record_count += 1
        if record_count == counter_file and df.shape[0] != 0:
            try:
                df["precio"] = df["precio"].fillna(0).astype(int)
            except:
                df["precio"] = 0
            try:
                df["año"] = np.where(df["año"].astype(str).str.endswith("días"), 0, df["año"])
                df["año"] = df["año"].fillna(0).astype(int)
            except:
                df["año"] = actual_year
            try:
                df["km"] = [x.rstrip(" km").replace(".","") for x in df["km"].astype(str)]
                df["km"] = df["km"].replace("nan",0).fillna(0).astype(int)
            except:
                df["km"] = 0
            df["dtm_etl"] = df["dtm_etl"].astype(str)
            df["id_ofert"] = df["id_ofert"].astype(str)
            df["barrio"] = df["barrio"].astype(str)
            df["ciudad"] = df["ciudad"].astype(str)
            df["departamento"] = df["departamento"].astype(str)

            #print(df.columns)
            if "a+-0" in df.columns:
                del df["a+-o"]
            df.rename(columns = {"versi>=n":"version", "transmisi>=n":"transmision", "tipo_de_carrocera":"tipo_de_carroceria",	
                                 "kil>=metros":"kilometros", "oltimo_dgito_de_la_placa":"ultimo_digito_de_la_placa",
                                 "con_cdegmara_de_reversa":"con_camara_de_reversa"}, inplace = True)

            #df.to_csv(f"temp/{actual_date}_{repetead + str(df.shape[0])}_{str(random.randint(1,10000))}.csv", sep="|", index=False)
            df.to_parquet(f"{ph.temp_car}/{actual_date}_{'H' + str(datetime.today().hour) + 'M' + str(datetime.today().minute)}_{repetead + str(df.shape[0])}.parquet", compression = "gzip")
            #func.pkl.Save(df, f"temp/{actual_date}_{repetead + str(df.shape[0])}_{str(random.randint(1,10000))}.pkl")

            
            df = pd.DataFrame()
            counter += 1
            record_count = 0  
            time.sleep(1)
        
        while (datetime.now() - time1).seconds > 20: #Funcional  60*2
            #print("Empieza descanso", datetime.now())
            #time.sleep(10)                             #Funcional 50
            time1 = datetime.now()
            #print("Termina descanso", time1)

        time.sleep(delay)
    print("Oferts Error Server 403:", error_server)
    print("Dimensión df:", df.shape)
    return df


def ReadTemp():
    import os
    files = os.listdir(ph.temp_car)
    #files = [pd.read_csv("temp/" + x, sep = "|") for x in files]
    df = pd.DataFrame()
    for x in files:
        df = pd.concat([df, pd.read_parquet(ph.temp_car + "/" + x, engine='pyarrow')], axis = 0)

    df.drop_duplicates("url", inplace=True)
    df = df[df["nombre"].notna()]
    
    def ClearFolder(carpeta):
        for archivo in os.listdir(carpeta):
            ruta_archivo = os.path.join(carpeta, archivo)
            try:
                if os.path.isfile(ruta_archivo):
                    os.remove(ruta_archivo)
            except Exception as e:
                pass
                #print(f'Error eliminando {ruta_archivo}: {e}')

    ClearFolder(ph.temp_car)
    cols_str = ["motor", "km", "año", "ultimo_digito_de_la_placa", "con_camara_de_reversa"]
    for i in cols_str:
        df[i] = df[i].astype(str)
    df.to_parquet(f"{ph.temp_car}/{actual_date}_{'H' + str(datetime.today().hour) + 'M' + str(datetime.today().minute)}_{str(df.shape[0])}.parquet", compression = "gzip")
    df = df.replace("nan", np.nan)
    print("Dimension df:", df.shape)
    return df


def TrainModel(df, save_model = True):
    cols_numeric = ["precio", "año", "km", "motor", "km_por_año"]
    for col in cols_numeric:
        df[col] = df[col].astype(float)
    df_train = df[["precio", "año", "km", "motor", "km_por_año", "transmision", "tipo_de_combustible", 
                   "tipo_de_carroceria", "puertas","marca", "modelo_agrup"]].dropna()
    df_train["precio"] = np.log(df_train["precio"].astype(float))

    X_train, X_test, y_train, y_test = models.TrainTestDummies(df = df_train, y='precio',
                                                                 dummies = ["marca", "modelo_agrup", "transmision", "tipo_de_combustible", 
                                                                            "tipo_de_carroceria", "puertas"])
    print("Dimensión train:", X_train.shape)
    print("Dimensión test:", X_test.shape)


    # Retrain model every week
    files_models = sorted(os.listdir(ph.models_pkl_car))
    files_models = [x for x in files_models if "_20" in x][-1]
    model = func.pkl.Load(f"{ph.models_pkl_car}/{files_models}")
    param_grid_actual = model.get_params()

    params = ['n_estimators', 'max_depth', 'min_child_weight', 'learning_rate', "eta","lambda","alpha","lambda", 'colsample_bytree',
              'subsample','booster', 'objective']
    param_grid_actual = {item: [param_grid_actual[item]] for item in params}


    r2_adj_actual = pd.read_csv(f"{ph.metrics_car}/metrics_xgboost.txt", sep = "|").sort_values("dtm_train", ascending = False)["r2_adj"][1]

    days_train = [1, 5, 6, 7 , 8, 14, 15, 16, 17, 18, 24] #

    if datetime.today().day in days_train:
        print("IMPORTANTE: Reentrenando modelo")
        param_grid_update = {'booster': ['gbtree'], 'objective': ['reg:squarederror'], "learning_rate":[0.01]}

        print("Grid actual")
        print(param_grid_actual)
        print("-"*100)

        for item, value in param_grid_actual.items():
            value = value[0]

            if item == "n_estimators":
                param_grid_update[item] = [max(0, value-30), value, value + 25]
            elif item in ["max_depth", "min_child_weight"]:
                param_grid_update[item] = [max(0, value-2), value, value + 2]
            elif item in ["eta", "lambda", "alpha"]:
                param_grid_update[item] = [max(0, value-0.05), value, min(1, value + 0.05)]
            elif item in ["colsample_bytree", "subsample"]:
                param_grid_update[item] = [max(0, value-0.1), value, min(1, value + 0.1)]

        for item, value in param_grid_update.items():
            if item in ["eta", "lambda", "alpha", "colsample_bytree", "subsample"]:
                param_grid_update[item] = [x for x in value if 0 <= x <= 1]
            elif item not in ["booster", "objective"]:
                param_grid_update[item] = [x for x in value if x >= 0]

        for item, value in param_grid_update.items():
            if len(value) == 2:
                if item == "n_estimators":
                    param_grid_update[item].append(max(value) + 25)
                elif item in ["max_depth", "min_child_weight"]:
                    param_grid_update[item].append(max(value) + 2)
                elif item in ["eta", "lambda", "alpha"]:
                    param_grid_update[item].append(min(1, max(value) + 0.05))
                elif item in ["colsample_bytree", "subsample"]:
                    param_grid_update[item].append(min(1, max(value) + 0.15))
        
        for item, value in param_grid_update.items():
            param_grid_update[item] = list(set(param_grid_update[item]))

        param_grid = param_grid_update
    else:
        param_grid = param_grid_actual

    # param_grid = {'objective': [ 'reg:squarederror'], 'booster': ['gbtree'],'colsample_bytree': [0.7],'learning_rate': [0.01],
    #               'max_depth': [8], 'min_child_weight': [14],'n_estimators': [400],'subsample': [1],'alpha': [0.5],'eta': [0],
    #               'lambda': [0.5]}

    print("-"*30)
    print("Grid de entrenamiento")
    print(param_grid)

    model, metrics_test = models.XGBoost(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, param_grid = param_grid, cv = 5, save = True).fit()

    # param_grid = {
    #             'layers': [[8]],  #, [4, 4], [2, 4, 6, 2], [4, 4, 4, 4]
    #             'activation': ['relu'],      
    #             'optimizer': ['adam'],         
    #             'epochs': [20],            
    #             'batch_size': [15]
    #         }
    # model, metrics_test = models.NN(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, param_grid = param_grid, cv = 2, save = True).fit()

    
    r2_adj_new = metrics_test["r2_adj"][0]
    if r2_adj_new > r2_adj_actual:
        print(f"IMPORTANTE: modelo actualizado: r2_adj_actual:{r2_adj_actual} < r2_adj_new:{r2_adj_new}")
        func.pkl.Save(model, f'{ph.models_pkl_car}/xgboost_{str(actual_date)}.pkl')
    else:
        print(f"IMPORTANTE: modelo no actualizado: r2_adj_actual:{r2_adj_actual} > r2_adj_new:{r2_adj_new}")
        param_grid = param_grid_actual
        
    #func.pkl.Save(model, f'{ph.models_pkl_car}/xgboost_{str("20240916")}.pkl')

    param_grid = model.get_params()
    params = ['n_estimators', 'max_depth', 'min_child_weight', 'learning_rate', "eta","lambda","alpha",
              "lambda", 'colsample_bytree','booster', 'objective']
    param_grid = {item: [param_grid[item]] for item in params}
    print(param_grid)

    return model

def Criterios(df):
    dim = df.shape[0]
    print("Registros:", dim)
    print("Excluyendo...")
    umbral_precio = np.percentile(df["precio"], 96)
    df = df[(df["precio"] > 0) & (df["precio"] <= umbral_precio)]
    print(f"Precio != 0 & < {round(umbral_precio,2)}:", df.shape[0])
    df = df[df["año"] != int(datetime.today().year)]
    print("Año != actual      :", df.shape[0])
    df = df[df["km"] > 0]
    print("Km > 0             :", df.shape[0])
    df = df[((df["motor"] > 0.5) & (df["motor"] < 8.1)) | (df["motor"].isnull())]
    print("Motor > 0.6 & < 8.1:", df.shape[0])
    df = df[df["antiguedad"] > 0]
    print("Antigúedad > 0     :", df.shape[0])
    df = df[df["transmision"].replace("nan",np.nan).notna()]
    print("transmision notna  :", df.shape[0])
    df = df[df["tipo_de_combustible"].replace("nan",np.nan).notna()]
    print("Combustible notna  :", df.shape[0])
    df = df[df["tipo_de_carroceria"].replace("nan",np.nan).notna()]
    print("Carroceria notna   :", df.shape[0])

    # Atributos homologados
    print("-"*30)
    print("Incluyendo atributos homologados")
    df = df[df["modelo"].isin(ho.models_homologue)]
    print("Modelos    :", df.shape[0])
    df = df[df["marca"].isin(ho.brands_homologue)]
    print("Marcas     :", df.shape[0])
    df = df[df["tipo_de_carroceria"].isin(ho.bodywork_homologue)]
    print("Carrocerias:", df.shape[0])
    df = df[df["transmision"].isin(ho.transmision_homologue)]
    print("Transmision:", df.shape[0])
    df = df[df["tipo_de_combustible"].isin(ho.fuel_homologue)]
    print("Combustible:", df.shape[0])

    print("*Total eliminados:", dim - df.shape[0], f"({round(((dim - df.shape[0]) / dim)*100, 2)}%)")

    return df

    
def Main(counter_file = 200, delay = 1, reprocess_oferts_urls = True): # Funcional delay = 1.5
    print("Iniciando Web Scrapping")
    time1 = datetime.now()

    save_model = True

    # Obtener número actual de ofertas
    #primer_pagina = "https://carros.mercadolibre.com.co/valle-del-cauca/cali/_Desde_49_ITEM*CONDITION_2230581_NoIndex_True"
    primer_pagina = "https://carros.mercadolibre.com.co/_Desde_49_ITEM*CONDITION_2230581_NoIndex_True" # Nacional
    response = requests.get(primer_pagina, headers=headers)
    response.raise_for_status()
    tree = html.fromstring(response.content)
    num_ofertas = int(tree.xpath('//*[@id="root-app"]/div/div[3]/aside/div[2]/span')[0].text_content().strip().replace(".","").replace(" resultados",""))
    print("Total ofertas:", num_ofertas)
    print("="*120)


    print("Bloque 1: obtener la url de cada oferta")
    #oferts_urls_temp = pd.read_parquet("data/urls.parquet", engine='pyarrow')
    #oferts_urls = oferts_urls_temp["url"]
    if reprocess_oferts_urls == True:
        oferts_urls = ObtainAllUrls()
    else:
        #oferts_urls = oferts_urls_temp
        print("Cargada oferts_urls de temporal")
    #oferts_urls = pd.Series(oferts_urls).sample(200).tolist()
    print("="*120)


    print("Bloque 2: obtener los atributos de cada oferta")
    try:
        df_temp = ReadTemp()
    except:
        df_temp = pd.DataFrame(columns = ["estado", "url"])
    oferts_urls_df = pd.read_parquet(f"{ph.data_car}/urls.parquet", engine='pyarrow')
    urls_complete_df = df_temp[df_temp["estado"] == 1]
    oferts_urls_df["id"] = oferts_urls_df["url"].str.split("-").str[1]
    urls_complete_df["id"] = urls_complete_df["url"].str.split("-").str[1]
    id_pendient_df = oferts_urls_df[~oferts_urls_df["id"].isin(urls_complete_df["id"])]
    urls_pendient = id_pendient_df["url"].tolist()
    print("Ofertas a ingestar:", len(urls_pendient))

    df = ScrapUrl(urls_pendient, delay = delay, counter_file = counter_file)
    print("="*120)


    print("Bloque 3: volver a consumir los registros con errores")
    for i in range(0,1):
        df = ReadTemp()
        df = pd.DataFrame(columns = ["estado", "url"])
        urls_complete = df[df["estado"] == 1]["url"]
        pendient = set(urls_pendient) - set(urls_complete)
        print("Registros correctos:", len(urls_complete), "|", round(len(urls_complete)/(len(urls_complete) + len(pendient))*100, 2), "%")
        print("Registros con error:", len(pendient), "|", round(len(pendient)/(len(urls_complete) + len(pendient))*100, 2), "%")
        df_error = ScrapUrl(pendient, delay = delay, counter_file = counter_file, repetead = True)


    print("Bloque 4: concatenado final de archivos")
    df = ReadTemp()
    df = df[df["estado"] == 1]
    print("="*120)


    print("Bloque 5: refinando atributos")
    df = RefineAtributes(df)
    print("-"*120)

    print("Correctos final:", df[df["estado"] == 1].shape[0], "|", round((df[df["estado"] == 1].shape[0]/df.shape[0])*100, 2), "%")
    print("Dimensión df:", df.shape)
    vars_duplicates = ["marca", "modelo", "precio", "año", "motor", "km", "transmision", "tipo_de_carroceria", "tipo_de_combustible", 
                       "barrio", "ciudad", "departamento"]
    print("Duplicados por atributos:", df[df.duplicated(vars_duplicates)].shape[0])
    print("Duplicados por url:", df[df.duplicated(["url"])].shape[0])
    print("Duplicados por id:", df[df.duplicated(["id_ofert"])].shape[0])
    df = df.drop_duplicates("url").drop_duplicates("id_ofert").drop_duplicates(vars_duplicates)
    print("Dimensión df sin duplicados:", df.shape)
    print("-"*120)
    pendient = set(oferts_urls) - set(df["url"])
    print("Final: Registros correctos:", len(urls_complete))
    print("Final: Registros con error (excluidos):", len(pendient))
    print("="*120)


    print("Bloque 6: imputación de datos faltantes en categoricas"); print("-"*50)
    df = models.ImputCategorics(df, "marca_modelo", "transmision", 70)
    df = models.ImputCategorics(df, "marca_modelo", "tipo_de_carroceria", 70)
    df = models.ImputCategorics(df, "marca_modelo", "tipo_de_combustible", 70)
    df = models.ImputCategorics(df, "marca_modelo", "puertas", 70)

    brands_ejecution = list(set(df["marca"].tolist()))
    models_ejecution = list(set(df["modelo"].tolist()))
    print("="*120)


    print("Bloque 7: imputando valores faltantes en cilindraje"); print("-"*50)
    df["motor"] = np.log(df["motor"])
    df = models.ImputRegression(df, y = 'motor', x = ['año', 'marca', 'modelo', 'tipo_de_combustible', 'transmision', 'tipo_de_carroceria', 
                                                    'puertas'], method = "xgboost")
    df["motor"] = round(np.exp(df["motor"]), 2)
    print("="*120)


    print("TENIENDO EN CUENTA DUMMYS DE NULOS")
    df["tipo_de_carroceria"] = df["tipo_de_carroceria"].fillna("NULO")
    df["tipo_de_combustible"] = df["tipo_de_combustible"].fillna("NULO")
    df["transmision"] = df["transmision"].fillna("NULO")

    print("Bloque 8: criterios de exclusión e inclusión"); print("-"*50)
    df = Criterios(df)
    print("="*120)


    print("Bloque 9: agrupacion de variables"); print("-"*50)
    df = models.ClassicationTreeGroup(df, y = "precio", x = "modelo", max_leaf_nodes=[30])
    print("="*120)

    df = df[['estado', 'nombre', 'precio', 'año', 'km', 'km_por_año', 'barrio', 'ciudad','departamento', 'fecha', 'url', 'dtm_etl', 
            'marca_modelo', 'marca', 'modelo', 'modelo_agrup', 'version','color', 'tipo_de_combustible', 'puertas', 'transmision', 
            'motor','tipo_de_carroceria', 'ultimo_digito_de_la_placa','con_camara_de_reversa']]
    
    save_model = True
    #if save_model == True:
    #df.to_csv(f"data/df_refine_{actual_date}.csv", sep = "|", index = False)
    df.to_parquet(f'{ph.data_car}/df_refine_{actual_date}.parquet', compression = "gzip")

    dim_models_group = df[["modelo", "modelo_agrup", "puertas"]].drop_duplicates()
    func.pkl.Save(dim_models_group, f'{ph.pkl_car}/dim_models_group.pkl')
    print("="*120)

    print("Bloque 10: entrenando modelo predictivo"); print("-"*50)
    model = TrainModel(df, save_model)
    print("="*120)

    print("---------- Precio ----------")
    print("Media:", round(df["precio"].mean(), 2))
    print("Mediana:", round(df["precio"].median(), 2))
    print("CV:", round(np.sqrt(df["precio"].var()) / df["precio"].mean() * 100, 2), "%")
    #print("Umbral:", round(umbral_precio, 3))
    print("="*120)

    print("Dimensión final:", df.shape)
    print(f"El proceso tardó: {datetime.now() - time1}")
    print("="*120)


    print("Marca")
    print("','".join(list(set(brands_ejecution) - set(ho.brands_homologue))))
    print("-"*120)
    print("Modelo")
    print("','".join(pd.Series(list(set(models_ejecution) - set(ho.models_homologue))).fillna("NULO")))
    print("-"*120)

    return df, model