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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')).replace("\\", "/")[:-5])
import models
import functions as func
import paths as ph

warnings.filterwarnings("ignore")
pd.options.display.max_columns = None

headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36 Edg/103.0.1264.49"}


actual_date = func.GetActualDate()
actual_year = datetime.today().year


def GetCarAtributes(
        url, 
        debug = "OFF"):
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

def ObtainAllUrls(delay = 1, debug = "OFF"):

    # Final filtrada
    marcas = ho.brands_url
    dim_ubicacion = ho.cities
    #dim_ubicacion = dim_ubicacion.sample()
    #dim_ubicacion = dim_ubicacion[dim_ubicacion["ciudad"] == "cali"]
    #dim_ubicacion = dim_ubicacion[dim_ubicacion["departamento"].isin(["valle-del-cauca", "bogota-dc", "antioquia"])]
    oferts_full = []

    departamentos = dim_ubicacion["departamento"].unique().tolist()
    for departamento in tqdm(departamentos, desc="Parseando ofertas por paginación"):

        ciudades = dim_ubicacion[dim_ubicacion["departamento"] == departamento]["ciudad"].unique().tolist()
        for ciudad in ciudades:
            if debug == "ON": print(ciudad)
            
            for marca in marcas:
                size = 0 # validacion
                for pag in range(49, 1969, 48):  # 48*n | El máximo de paginación por categoría es de 42, 1969 vehículos
                    url = f"https://motos.mercadolibre.com.co/{marca}/{departamento}/{ciudad}/_Desde_{pag}_ITEM*CONDITION_2230581_NoIndex_True"
                    #print(url)
                    response = requests.get(url, headers=headers)
                    content = response.content.decode('utf-8')
                    href_ofert = '//*[@id="root-app"]/div/div[3]/section/ol/li[1]/div'
                    
                    if "Escribe en el buscador lo que quieres encontrar." not in content and \
                        'No hay publicaciones que coincidan con tu búsqueda.' not in content:
                        tree = html.fromstring(content)
                        _urls_pag = []
                        elements = tree.xpath('//*[@id="root-app"]/div/div[3]/section/ol')
                        for element in elements:
                            items = element.xpath('.//a[@href]')
                            for item in items:
                                _urls_pag.append(item.get("href"))
                        oferts_full.append(_urls_pag)
                        time.sleep(delay)
                        size += len(_urls_pag)
                    else:
                        base_url = "/".join(url.split("/")[:6])
                        response = requests.get(base_url, headers=headers)
                        tree = html.fromstring(response.content)
                        _urls_pag = []
                        elements = tree.xpath('//*[@id="root-app"]/div/div[3]/section/ol')
                        for element in elements:
                            items = element.xpath('.//a[@href]')
                            for item in items:
                                _urls_pag.append(item.get("href"))
                        oferts_full.append(_urls_pag)
                        time.sleep(delay)
                        size += len(_urls_pag)
                        break
                if debug == "ON": print(marca, size)
        time.sleep(7)

    oferts_full = [i for sublist in oferts_full for i in sublist]
    oferts_full = list(set(oferts_full))
    pd.DataFrame({"url":oferts_full}).to_parquet(f"{ph.data_bike}/urls.parquet", compression = "gzip")
    print("Ofertas obtenidas:", len(oferts_full))

    return oferts_full


def ReadUrlsParquet():
    oferts_full = pd.read_parquet(f"{ph.data_bike}/urls.parquet", engine='pyarrow')["url"].tolist()
    oferts_full = [i for sublist in oferts_full for i in sublist]
    oferts_full = list(set(oferts_full))
    print(len(oferts_full))

    return oferts_full


def ScrapUrl(oferts_full, 
             delay = 1,
             counter_file = 200,
             repetead = False,
             debug = "OFF"
             ):
    df = pd.DataFrame()
    error_server = 0
    counter = 1
    change_time = False
    from datetime import datetime
    time1 = datetime.now()
    record_count = 0
    repetead = "_R" if repetead == True else "C" # C: registros correctos, R: registros que se volvieron a ingestar debido a error en andes-table

    for oferta_temp in tqdm(oferts_full, desc="Obteniendo variables"):
        if debug == "ON": print(oferta_temp)
        try:
            df_oferta_temp, error_server_temp = GetCarAtributes(oferta_temp, debug = "OFF")
        except:
            df_oferta_temp = pd.DataFrame(columns = ["url", "estado"])
            error_server_temp = False
        df_oferta_temp = df_oferta_temp[df_oferta_temp["estado"] == 1]
        df = pd.concat([df, df_oferta_temp], axis = 0)

        if "marca" in df_oferta_temp.columns:
            if debug == "ON": print("correcto")
        else:
            if debug == "ON": print("incorrecto")

        error_server += error_server_temp
        if error_server > len(oferts_full)*0.02:
            save_model = False
            raise ValueError("❌ ERROR: Error Server > 2% registers:", error_server)
            break

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
            df.rename(columns = {}, inplace = True)

            df.to_parquet(f"{ph.temp_bike}/{actual_date}_{'H' + str(datetime.today().hour) + 'M' + str(datetime.today().minute)}_{repetead + str(df.shape[0])}.parquet", compression = "gzip")

            df = pd.DataFrame()
            counter += 1
            record_count = 0  
            time.sleep(1)
        
        while (datetime.now() - time1).seconds > 50: #Funcional  60*2
            time.sleep(5)                            #Funcional 50
            time1 = datetime.now()

        time.sleep(delay)
    print("Oferts Error Server 403:", error_server)
    print("Dimensión df:", df.shape)
    return df


def ReadTemp():
    import os
    files = os.listdir(ph.temp_bike)
    df = pd.DataFrame()
    for x in files:
        df = pd.concat([df, pd.read_parquet(ph.temp_bike + "/" + x)], axis = 0)

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

    ClearFolder(ph.temp_bike)
    cols_str = ["cilindrada", "km", "año"]
    for i in cols_str:
        df[i] = df[i].astype(str)
    df.to_parquet(f"{ph.temp_bike}/{actual_date}_{'H' + str(datetime.today().hour) + 'M' + str(datetime.today().minute)}_{str(df.shape[0])}.parquet", compression = "gzip")
    df = df.replace("nan", np.nan)
    print("Dimension df:", df.shape)
    return df


def TrainModel(df, save_model = True):
    cols_numeric = ["precio", "año", "km", "4_tiempos", "cilindrada", "km_por_año"]
    for col in cols_numeric:
        df[col] = df[col].astype(float)
    df_train = df[["precio", "año", "km", "cilindrada", "tipo_de_moto", "4_tiempos", 
                   "km_por_año","marca", "modelo_agrup"]].dropna()
    df_train["precio"] = np.log(df_train["precio"].astype(float))

    X_train, X_test, y_train, y_test = models.TrainTestDummies(df = df_train, y='precio',
                                                               dummies = ["marca", "modelo_agrup", "tipo_de_moto"])
    print("Dimensión train:", X_train.shape)
    print("Dimensión test:", X_test.shape)


    # Retrain model every week
    # files_models = sorted(os.listdir(ph.models_pkl_bike))
    # files_models = [x for x in files_models if "_20" in x][-1]
    # model = func.pkl.Load(f"{ph.models_pkl_bike}/{files_models}")
    # param_grid_actual = model.get_params()

    # params = ['n_estimators', 'max_depth', 'min_child_weight', 'learning_rate', "eta","lambda","alpha","lambda", 'colsample_bytree',
    #           'subsample','booster', 'objective']
    # param_grid_actual = {item: [param_grid_actual[item]] for item in params}


    # r2_adj_actual = pd.read_csv(f"{ph.metrics_bike}/metrics_xgboost.txt", sep = "|").sort_values("dtm_train", ascending = False)["r2_adj"][1]

    # days_train = [0]#[1, 8, 14, 15, 16, 17, 18, 24] #

    # if datetime.today().day in days_train:
    #     print("IMPORTANTE: Reentrenando modelo")
    #     param_grid_update = {'booster': ['gbtree'], 'objective': ['reg:squarederror'], "learning_rate":[0.01]}

    #     print("Grid actual")
    #     print(param_grid_actual)
    #     print("-"*100)

    #     for item, value in param_grid_actual.items():
    #         value = value[0]

    #         if item == "n_estimators":
    #             param_grid_update[item] = [max(0, value-30), value, value + 25]
    #         elif item in ["max_depth", "min_child_weight"]:
    #             param_grid_update[item] = [max(0, value-2), value, value + 2]
    #         elif item in ["eta", "lambda", "alpha"]:
    #             param_grid_update[item] = [max(0, value-0.05), value, min(1, value + 0.05)]
    #         elif item in ["colsample_bytree", "subsample"]:
    #             param_grid_update[item] = [max(0, value-0.1), value, min(1, value + 0.1)]

    #     for item, value in param_grid_update.items():
    #         if item in ["eta", "lambda", "alpha", "colsample_bytree", "subsample"]:
    #             param_grid_update[item] = [x for x in value if 0 <= x <= 1]
    #         elif item not in ["booster", "objective"]:
    #             param_grid_update[item] = [x for x in value if x >= 0]

    #     for item, value in param_grid_update.items():
    #         if len(value) == 2:
    #             if item == "n_estimators":
    #                 param_grid_update[item].append(max(value) + 25)
    #             elif item in ["max_depth", "min_child_weight"]:
    #                 param_grid_update[item].append(max(value) + 2)
    #             elif item in ["eta", "lambda", "alpha"]:
    #                 param_grid_update[item].append(min(1, max(value) + 0.05))
    #             elif item in ["colsample_bytree", "subsample"]:
    #                 param_grid_update[item].append(min(1, max(value) + 0.15))
        
    #     for item, value in param_grid_update.items():
    #         param_grid_update[item] = list(set(param_grid_update[item]))

    #     param_grid = param_grid_update
    # else:
    #     param_grid = param_grid_actual

    param_grid = {'objective': [ 'reg:squarederror'], 'booster': ['gbtree'],
                  'colsample_bytree': [0.4, 0.7, 0.8],
                  'learning_rate': [0.01, 0.001],
                  'max_depth': [3, 5, 8, 12], 'min_child_weight': [3, 5, 10, 14],
                  'n_estimators': [150, 250, 400],'subsample': [0.6,  0.85, 1],
                  'alpha': [0.3, 0.8],'eta': [0.3, 0.8],
                  'lambda': [0.3, 0.8]}

    print("-"*30)
    print("Grid de entrenamiento")
    print(param_grid)

    model, metrics_test = models.XGBoost(X_train = X_train, X_test = X_test, y_train = y_train, 
                                         y_test = y_test, param_grid = param_grid, cv = 5, save = True).fit()

    # param_grid = {
    #             'layers': [[8]],  #, [4, 4], [2, 4, 6, 2], [4, 4, 4, 4]
    #             'activation': ['relu'],      
    #             'optimizer': ['adam'],         
    #             'epochs': [20],            
    #             'batch_size': [15]
    #         }
    # model, metrics_test = models.NN(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, param_grid = param_grid, cv = 2, save = True).fit()

    
    r2_adj_actual = 0
    param_grid_actual = {}
    r2_adj_new = metrics_test["r2_adj"][0]
    if r2_adj_new > r2_adj_actual:
        print(f"IMPORTANTE: modelo actualizado: r2_adj_actual:{r2_adj_actual} < r2_adj_new:{r2_adj_new}")
        func.pkl.Save(model, f'{ph.models_pkl_bike}/xgboost_{str(actual_date)}.pkl')
    else:
        print(f"IMPORTANTE: modelo no actualizado: r2_adj_actual:{r2_adj_actual} > r2_adj_new:{r2_adj_new}")
        param_grid = param_grid_actual
        
    #func.pkl.Save(model, f'{ph.models_pkl_bike}/xgboost_{str("20240916")}.pkl')

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
    df = df[((df["cilindrada"] > 60) & (df["cilindrada"] < 3000)) | (df["cilindrada"].isnull())]
    print("cilindrada => 60 & < 3000:", df.shape[0])
    df = df[df["antiguedad"] > 0]
    print("Antigúedad > 0     :", df.shape[0])

    # Atributos homologados
    print("-"*30)
    print("Incluyendo atributos homologados")
    # df = df[df["modelo"].isin(ho.models_all)]
    # print("Modelos    :", df.shape[0])
    # df = df[df["marca"].isin(ho.brands_homologue)]
    # print("Marcas     :", df.shape[0])

    print("*Total eliminados:", dim - df.shape[0], f"({round(((dim - df.shape[0]) / dim)*100, 2)}%)")

    return df

    
def Main(counter_file = 200, delay = 1, reprocess_oferts_urls = True, debug = "OFF"): # Funcional delay = 1.5
    print("Iniciando Web Scrapping")
    time1 = datetime.now()

    save_model = True

    # Obtener número actual de ofertas
    #primer_pagina = "https://carros.mercadolibre.com.co/valle-del-cauca/cali/_Desde_49_ITEM*CONDITION_2230581_NoIndex_True"
    primer_pagina = "https://motos.mercadolibre.com.co/_Desde_49_ITEM*CONDITION_2230581_NoIndex_True" # Nacional
    response = requests.get(primer_pagina, headers=headers)
    response.raise_for_status()
    tree = html.fromstring(response.content)
    num_ofertas = int(tree.xpath('//*[@id="root-app"]/div/div[3]/aside/div[2]/span')[0].text_content().strip().replace(".","").replace(" resultados",""))
    print("Total ofertas:", num_ofertas)
    print("="*120)


    print("Bloque 1: obtener la url de cada oferta")
    if reprocess_oferts_urls == True:
        oferts_urls = ObtainAllUrls()
    else:
        oferts_urls = pd.read_parquet(f"{ph.data_bike}/urls.parquet", engine='pyarrow')
        oferts_urls = oferts_urls["url"]
        print("Cargada oferts_urls de temporal")
    #oferts_urls = pd.Series(oferts_urls).sample(200).tolist()
    print("="*120)


    print("Bloque 2: obtener los atributos de cada oferta")
    try:
        df_temp = ReadTemp()
    except:
        df_temp = pd.DataFrame(columns = ["estado", "url"])
    oferts_urls_df = pd.read_parquet(f"{ph.data_bike}/urls.parquet", engine='pyarrow').drop_duplicates()
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
    vars_duplicates = ["marca", "modelo", "precio", "año", "cilindrada", "km", "barrio", "ciudad", "departamento"]
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
    #df = models.ImputCategorics(df, "marca_modelo", "transmision", 70)


    brands_ejecution = list(set(df["marca"].tolist()))
    models_ejecution = list(set(df["modelo"].tolist()))
    print("="*120)


    print("Bloque 7: imputando valores faltantes en cilindraje"); print("-"*50)
    df["cilindrada"] = np.log(df["cilindrada"])
    df = models.ImputRegression(df, y = 'cilindrada', x = ['año', 'marca', 'modelo'], method = "xgboost")
    df["cilindrada"] = round(np.exp(df["cilindrada"]), 2)
    print("="*120)


    print("TENIENDO EN CUENTA DUMMYS DE NULOS")


    print("Bloque 8: criterios de exclusión e inclusión"); print("-"*50)
    df = Criterios(df)
    print("="*120)


    print("Bloque 9: agrupacion de variables"); print("-"*50)
    df = models.ClassicationTreeGroup(df, y = "precio", x = "modelo", max_leaf_nodes=[30])
    print("="*120)

    df = df[['estado', 'nombre', 'precio', 'año', 'km', 'km_por_año', "tipo_de_moto", "4_tiempos", 'barrio', 'ciudad','departamento',
             'fecha', 'url', 'dtm_etl', 'marca_modelo', 'marca', 'modelo', 'modelo_agrup','cilindrada']]
    
    save_model = True
    #if save_model == True:
    #df.to_csv(f"data/df_refine_{actual_date}.csv", sep = "|", index = False)
    df.to_parquet(f'{ph.data_bike}/df_refine_{actual_date}.parquet', compression = "gzip")

    dim_models_group = df[["modelo", "modelo_agrup"]].drop_duplicates()
    func.pkl.Save(dim_models_group, f'{ph.pkl_bike}/dim_models_group.pkl')
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
    print("','".join(pd.Series(list(set(models_ejecution) - set(ho.models_all))).fillna("NULO")))
    print("-"*120)

    return df, model

def CompleteAtributes(df):

    df["modelo"] = df["modelo"].str.lower().replace({'mt 07':"mt07", 'mt 01':'mt01', "mt 09":"mt09", "mt 15":"mt15", 
                                                    "mt 03":"mt03"})

    def find_brand2w(row, models):
        model = row["modelo"].lower()
        if model in models:
            model_final =  model.capitalize()
        else:
            model_final =  "PENDIENTE"

        if model_final == "PENDIENTE":
            onew_in_model = [x for x in models if x in model]
            if onew_in_model != []:
                model_final = onew_in_model[0].capitalize()
        return model_final
        
    df['modelo2w'] = df.apply(find_brand2w, models=ho.models["2w"], axis=1)

    def find_brand1w(row, models):
        tokens = row["modelo"].lower().split() 
        if row["modelo2w"] == "PENDIENTE":
            for token in tokens:
                if token in models:
                    return token.capitalize()
        else:
            return "PENDIENTE"

    df['modelo1w'] = df.apply(find_brand1w, models=ho.models["1w"], axis=1)
    df["modelo_homo"] = np.where(df["modelo2w"] != "PENDIENTE", df["modelo2w"], df["modelo1w"])
    df["modelo_homo"] = df["modelo_homo"].fillna("PENDIENTE").replace({"Nmax":"N-max", "X max":"X-max"})
    del df["modelo1w"], df["modelo2w"]

    def asign_abrev(row):
        modelo = row["modelo_homo"].lower().split()
        index = 0
        for j in modelo:
            if j in ho.abreviaturas:
                abrev = [x for x in ho.abreviaturas if j == x][0]
                abrev = abrev.upper()
                modelo[index] = abrev
            else:
                modelo[index] = modelo[index].capitalize()
            index += 1
        return " ".join(modelo)
        
    df["modelo_homo"] = df.apply(asign_abrev, axis = 1)
    df["modelo_homo"] = df["modelo_homo"].replace({'200 ns':"NS 200", '160 nc':"NC 160"})

    c = df[df['modelo_homo'] == "Pendiente"].shape[0]
    print("Nulos en modelo homologado:", c, c/df.shape[0]*100)

    return df



def RefineAtributes(df, debug = "OFF"):
    """
    Función que refina y homologa los atributos del vehículo
    """
    time1 = time.time()

    df = CompleteAtributes(df)
    df = df.replace("nan", np.nan).replace("missing", np.nan)

    df["modelo"] = df["modelo_homo"]
    del df["modelo_homo"]

    def _refine_cilindrada(x):
        x = func.remove_letters(x.replace(" CC",""))
        return x
    df["cilindrada"] = [_refine_cilindrada(x) for x in df["cilindrada"]]
    df["cilindrada"] = df["cilindrada"].replace({"":np.nan, "0":np.nan})
    df["cilindrada"] = pd.to_numeric(df["cilindrada"], errors = "coerce")

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

    # vars_null = ["tipo_de_carroceria", "tipo_de_combustible", "transmision", "color"]
    # for col in vars_null:
    #     df[col] = df[col].fillna("missing")

    df["nombre"] = [unidecode(x.lower()) for x in df["nombre"].astype(str)]
    df["nombre"] = df["nombre"].replace({"nan":np.nan})


    df["año"] = df["año"].apply(lambda x: func.convert_to_numeric(x, "int"))
    df["año"] = df["año"].fillna(actual_year).astype(int)
    
    df["antiguedad"] = np.where(actual_year - df["año"] == 0, 1, actual_year - df["año"])

    df["4_tiempos"] = np.where(df["motor"] == "4 tiempos", 1, 0)
    del df["motor"]

    df["precio"] = round(df["precio"]/1000000, 3)

    df["marca_modelo"] = df["marca"] + " " + df["modelo"]

    df["km_por_año"] = round(df["km"].fillna(0).astype(float) / df["antiguedad"], 2)
    #df["sk_id"] = [i for i in range(0, df.shape[0])]

    df["año"] = df["año"].fillna(actual_year).astype(int)
    df["km"] = df["km"].fillna(0).astype(float).replace(0,np.nan)
    df["antiguedad"] = df["antiguedad"].fillna(0).astype(int).replace(0,np.nan)
    #df["ultimo_digito_de_la_placa"] = df["ultimo_digito_de_la_placa"].replace(None,999).fillna(999).astype(float).replace(999,np.nan)

    # Imputando tipo de moto
    df["tipo_de_moto"] = df["tipo_de_moto"].replace({"Calle":"Street","Scooters":"Scooter", 
                                  "Deportivas":"Deportiva",
                                  "Cuatrimotos":"Cuatrimoto",
                                  'Motocarros':"Motocarro", 'Sport':"Deportiva","Urbana":"Street", "Mini motos":"Mini moto",
                                  "Cross":"Motocross", "Urbanas":"Street", "Ciudad":"Street", 'Cruiser':"Crucero",
                                  'Sport touring':"Touring", "Todo terreno":"Enduro" ,"Clásico":"Clásica",
                                  "Clasica":"Clásica", 'Motocarguero':"Motocarro", 'Motos electricas':"Eléctrica",
                                  "Neked":"Naked","Multi proposito":"Doble propósito","Multipropósito":"Doble propósito",
                                  "Señoritera":"Scooter", 'Moto electrica':"Eléctrica", 'Electrica':"Eléctrica", 
                                  'Touring':"Turismo"})

    df_types = df[df["tipo_de_moto"].isin(ho.moto_types)]
    dicc_types = pd.DataFrame()
    marca_modelo = df_types["marca_modelo"].unique()
    for i in marca_modelo:
        ax = df_types[df_types["marca_modelo"] == i]
        type_model = ax["tipo_de_moto"].value_counts().reset_index()
        type_model["porc"] = round(type_model["tipo_de_moto"]/ax.shape[0]*100, 2)
        type_model["marca_modelo"] = i
        del type_model["tipo_de_moto"]
        dicc_types = pd.concat([dicc_types, type_model], axis = 0)
    dicc_types = dicc_types[dicc_types["porc"] >= 45].sort_values("porc",ascending=False).drop_duplicates("marca_modelo")
    dicc_types.rename(columns = {"index":"tipo_de_moto_imput"}, inplace = True)
    dicc_types = dicc_types[["marca_modelo", "tipo_de_moto_imput"]]

    # Merge para medir precision
    df_types = pd.merge(df_types , dicc_types, on = "marca_modelo", how = "left")
    c = df_types["tipo_de_moto_imput"].isnull().sum()
    print("total", df_types.shape[0], "| nulos", c, c/df_types.shape[0])

    metrics = df_types[["tipo_de_moto", "tipo_de_moto_imput"]].dropna()
    c = metrics[metrics["tipo_de_moto"] == metrics["tipo_de_moto_imput"]].shape[0]
    print(f"Precision: {round(c/metrics.shape[0], 4)}")

    # Merge final
    df = pd.merge(df , dicc_types, on = "marca_modelo", how = "left")
    df["tipo_de_moto"] = df["tipo_de_moto_imput"]
    del df["tipo_de_moto_imput"]


    # Filtros varios
    df = df[df["año"] != 0]
    df = df[df["km"] <= 600000]

    if 'aa+-o' in df.columns:
        df.drop(columns = ['aa+-o', 'kila3metros'], inplace = True)

    if debug == "ON": print(f"RefineAtributes: {time.time() - time1:.2f} seg")
    time1 = time.time()

    return df