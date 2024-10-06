import gradio as gr
import pandas as pd
import numpy as np
import warnings
import joblib
import os, sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')).replace("\\", "/"))
import functions as func

warnings.filterwarnings("ignore")

#df = pd.read_csv("data/df_refine.csv", sep = "|")
#df = pd.read_parquet("data/df_refine.parquet")

paths = sorted(os.listdir("data"))
paths = [x for x in paths if "_20" in x][-1]
df = pd.read_parquet("data/" + paths)
#df = pd.read_parquet("data/df_refine.parquet")

date = sorted(df["dtm_etl"].unique().astype(str).tolist(), reverse=True)[1]

transmision_options = sorted(df["transmision"].unique().astype(str).tolist())
if "NULO" in transmision_options: transmision_options.remove("NULO")
if "nan" in transmision_options: transmision_options.remove("nan")
carroceria_options = sorted(df["tipo_de_carroceria"].unique().astype(str).tolist())
if "NULO" in carroceria_options: carroceria_options.remove("NULO")
if "nan" in carroceria_options: carroceria_options.remove("nan")
combustible_options = sorted(df["tipo_de_combustible"].unique().astype(str).tolist())
if "NULO" in combustible_options: combustible_options.remove("NULO")
if "nan" in combustible_options: combustible_options.remove("nan")
marcas_options = sorted(df["marca"].unique().astype(str).tolist())
if "nan" in marcas_options: marcas_options.remove("nan")
ciudad_options = sorted(df["ciudad"].unique().astype(str).tolist())
if "nan" in ciudad_options: ciudad_options.remove("nan")

n_estimation = 0
global_predict = 0
error_predict = False
def Predict(marca, modelo, transmision, tipo_de_carroceria, tipo_de_combustible, año, motor, km, km_check):
    files_models = sorted(os.listdir("models_pkl"))
    files_models = [x for x in files_models if "_20" in x][-1]
    model = joblib.load("models_pkl/" + files_models)  
    dim_models = func.pkl.Load("pkl/dim_models_group.pkl")[["modelo", "modelo_agrup"]].drop_duplicates()
    dim_doors = func.pkl.Load("pkl/dim_models_group.pkl")[["modelo", "puertas"]].drop_duplicates()

    modelo_orig = modelo
    modelo = dim_models[dim_models["modelo"] == modelo]["modelo_agrup"].unique()[0]
    puertas = dim_doors[dim_doors["modelo"] == modelo]["puertas"].unique().tolist()
    if puertas == []:
        puertas = "4_5"
    else:
        puertas = puertas[0]

    if km_check == True:
        km = df[(df["marca"] == marca) & (df["modelo"] == modelo) & (df["año"] == modelo)]["km"].mean()
        if np.isnan(km) == True:
            km = df[(df["marca"] == marca) & (df["modelo"] == modelo)]["km"].mean()


    df_new = pd.DataFrame({"año":[año], "km":[km], "motor":[motor], "km_por_año":[km/(datetime.today().year-año)], 
                           "transmision":[transmision], "tipo_de_combustible":[tipo_de_combustible],"marca":[marca], 
                           "modelo_agrup":[modelo], "puertas":[puertas], "tipo_de_carroceria":[tipo_de_carroceria]})
    df_new = pd.get_dummies(df_new, columns = ["transmision", "tipo_de_combustible", "marca", "modelo_agrup",
                                               "puertas", "tipo_de_carroceria"]).replace(True, 1)
    features = model.feature_names_in_.tolist()
    df_predict = pd.DataFrame(columns = features)
    df_predict.loc[0] = 0
    for col in df_new.columns:
        if col in features:
            df_predict[col] = df_new[col]
    predict = int(str(int(np.exp(model.predict(df_predict))*1000000))[:-3] + "000") 
    global global_predict
    global_predict = predict
    predict = f"${predict:,}".replace(",", ".") + " millones de pesos"
    
    global n_estimation
    n_estimation = df[(df["marca"] == marca) & (df["modelo"] == modelo_orig)].shape[0]

    bodywork_verify_ = df[(df["marca"] == marca) & (df["modelo"] == modelo_orig) & (df["tipo_de_carroceria"] == tipo_de_carroceria)].shape[0]
    bodywork_real_= ", ".join(df[(df["marca"] == marca) & (df["modelo"] == modelo_orig)]["tipo_de_carroceria"].unique().tolist())
    bodywork_real_ = bodywork_real_.replace("NULO", "")
    
    transmision_verify_ = df[(df["marca"] == marca) & (df["modelo"] == modelo_orig) & (df["transmision"] == transmision)].shape[0]
    transmision_real_= ", ".join(df[(df["marca"] == marca) & (df["modelo"] == modelo_orig)]["transmision"].unique().tolist())
    transmision_real_ = transmision_real_.replace("NULO", "")
    
    fuel_verify_ = df[(df["marca"] == marca) & (df["modelo"] == modelo_orig) & (df["tipo_de_combustible"] == tipo_de_combustible)].shape[0]
    fuel_real_= ", ".join(df[(df["marca"] == marca) & (df["modelo"] == modelo_orig)]["tipo_de_combustible"].unique().tolist())
    fuel_real_ = fuel_real_.replace("NULO", "")


    global error_predict
    marca_modelo =  df[(df["marca"] == marca) & (df["modelo"] == modelo_orig)]["marca_modelo"].unique().tolist()[0]
    if bodywork_verify_ == 0:
        error_predict = True
        return f"Tipo de carroceria no disponible para {marca_modelo}, los disponibles son: {bodywork_real_}"
    elif transmision_verify_ == 0:
        error_predict = True
        return f"Transmisión no disponible para {marca_modelo}, los disponibles son: {transmision_real_}"
    elif fuel_verify_ == 0:
        error_predict = True
        return f"Tipo de combustible no disponible para {marca_modelo}, los disponibles son: {fuel_real_}"
    else:
        return predict
    

def SimulateCredit(n, cuota_inicial, EA):
    global global_predict
    EA = EA/100

    tasa_EM = (1+EA)**(1/12) - 1
    cuota_mes = ((global_predict - cuota_inicial)*tasa_EM) / (1-(1+tasa_EM)**(-n))
    
    return f"${int(cuota_mes):,}".replace(",", ".") + " pesos"


def UpdateBrand(brand):
    models = sorted(df[df["marca"] == brand]["modelo"].unique().astype(str).tolist())
    if "nan" in models: models.remove("nan")
    return gr.Dropdown(choices=models, interactive=True)

def UpdateBodyWork(model):
    bodys = sorted(df[df["modelo"] == model]["tipo_de_carroceria"].unique().astype(str).tolist())
    if "nan" in bodys: bodys.remove("nan")
    return gr.Dropdown(choices=bodys, interactive=True)

def plot_median_price_by_brand(car_model = None):
    global df
    import matplotlib.pyplot as plt
    if car_model != None:
        df = df[df['modelo'] == car_model]
    top_10_models = df['marca_modelo'].value_counts().nlargest(7).index
    df_top_10 = df[df['marca_modelo'].isin(top_10_models)]
    median_prices = df_top_10.groupby('marca_modelo')['precio'].median().reset_index().sort_values("precio", ascending = True).head(7)
    median_prices = median_prices.sort_values("precio", ascending = False)
    only_medians = median_prices.copy()
    median_prices.index = median_prices["marca_modelo"]
    fig, ax = plt.subplots()
    median_prices.plot(kind='barh', ax=ax, color='#60B0F0', figsize=(10,3.5), xlim=[0, max(median_prices["precio"])*1.1])

    index = list(range(0,len(only_medians)))
    for i in index:
        value = only_medians.iloc[i]["precio"]
        ax.text(value*1.01, i, f'${round(float(value), 2)} mill.', va='center', ha='left', fontsize=10, color='black')

    #ax.xaxis.set_visible(False)  # Oculta los números del eje x
    #ax.yaxis.set_ticks([])       # Oculta los números del eje y

    plt.subplots_adjust(left=0.25, right=0.85, top=0.9, bottom=0.15)
    ax.set_title('Modelos mas económicos')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend().set_visible(False)

    return fig



def plot_frequent_by_brand():
    import matplotlib.pyplot as plt
    top_10_models = df['marca_modelo'].value_counts().nlargest(10).index
    df_top_10 = df[df['marca_modelo'].isin(top_10_models)]
    df_top_10["cont"] = 1
    median_prices = df_top_10.groupby('marca_modelo')['cont'].sum().reset_index().sort_values("cont", ascending = True).head(10)
    median_prices = median_prices.sort_values("cont", ascending = True)
    only_medians = median_prices.copy()
    median_prices.index = median_prices["marca_modelo"]
    median_prices["cont"] = round(median_prices["cont"] / df.shape[0] *100, 1)
    fig, ax = plt.subplots()

    median_prices.plot(kind='barh', ax=ax, color='#60B0F0', figsize=(11,4), xlim=[0, max(median_prices["cont"])*1.1])

    index = list(range(0,len(only_medians)))
    for i in index:
        value = median_prices.iloc[i]["cont"]
        ax.text(value*1.01, i, f'{value} %', va='center', ha='left', fontsize=10, color='black')
    plt.subplots_adjust(left=0.25, right=0.8, top=0.9, bottom=0.15)
    ax.set_title('Participación en el mercado de plataformas digitales por modelo')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend().set_visible(False)

    return fig

with gr.Blocks(theme=gr.themes.Soft(), css="styles.css") as demo: #gr.themes.Soft(), gr.themes.Monochrome()
    import random
    c = random.randint(4000, 60000)
    c = round(c/(60*60), 1)
    c = str(np.where(str(c)[-1] == "0", str(c)[0], str(c)))
    car_sample = df[df["precio"] >= 12].sample(1)

    
    
    gr.Markdown(
        """
        <div style="text-align: center; max-width: 8000; margin: 50px auto 60px auto;">
            <h1 style="font-size: 2.7rem; font-weight: 700; margin-bottom: 2rem; margin-bottom: 2rem; display: contents;">Precio de venta de autos usados en Colombia</h1>
            <p style="font-size: 1rem; margin-bottom: 1.5rem;"></p>
        </div>
        """
    )

    gr.Markdown(f"### ⏰ Datos actualizados al {date}.")
    #gr.Markdown(f'### 🚗 Último vehículo ofertado hace {c} horas por ${car_sample["precio"].tolist()[0]} millones: {car_sample["marca_modelo"].tolist()[0]} | {int(car_sample["km"].tolist()[0]):,} km | Año del modelo: {car_sample["año"].tolist()[0]} | {round(car_sample["motor"].tolist()[0], 1)} litros.')

    gr.Markdown("")
    
    gr.Markdown("## Marca y modelo")
    with gr.Row():
        marca = gr.Dropdown(choices=sorted(df["marca"].unique().astype(str).tolist()), label="Selecciona una marca", value="Selecciona una marca")
        modelo = gr.Dropdown(label="Selecciona un modelo", choices=[])
    marca.change(UpdateBrand, inputs=[marca], outputs=[modelo])

    gr.Markdown("## Variables técnicas")
    with gr.Row():
        transmision = gr.Dropdown(choices=transmision_options, label="Selecciona un tipo de transmisión", value="Automática")
        tipo_de_carroceria = gr.Dropdown(choices=carroceria_options, label="Selecciona un tipo de carrocería", value="Camioneta")
        tipo_de_combustible = gr.Dropdown(choices=combustible_options, label="Selecciona un tipo de combustible", value="Gasolina")
        # motor = gr.Number(label="Digita el cilindraje en litros", value=2, precision=2, minimum=0.5, maximum=8)
        motor = gr.components.Slider(label = "Digita el cilindraje en litros", minimum = 0.6, maximum= 5, step = 0.1, value=2)
    #modelo.change(UpdateBrand, inputs=[modelo], outputs=[tipo_de_carroceria])

    gr.Markdown("## Variables de uso")
    with gr.Row():
        km_check = False
        año = gr.components.Slider(label = "Digita el año del modelo", minimum = 1980, maximum= datetime.today().year, step = 1, value=2020)
        km = gr.components.Slider(label = "Digita los kilómetros recorridos", minimum = 0, maximum= 500000, step = 10000, value=50000)
        km_check = gr.Checkbox(label="Ignorar kilómetros (recomendado si vas a comprar y no tienes definido un kilómetraje específico)")

    # gr.Markdown("## Ubicación")
    # with gr.Row():
    #     ciudad = gr.Dropdown(choices=ciudad_options, label="Selecciona la ubicación del vehículo")
    
    b1 = gr.Button("Calcular precio del auto", elem_classes="button")
    gr.Markdown("---")

    gr.Markdown(
    """
        <div style="text-align: left; max-width: 8000; margin: 0 auto;">
            <h2 style="font-size: 2.2rem; font-weight: 700; margin-bottom: 1rem; margin-bottom: 1.5rem; display: contents;">💲 Precio estimado de venta</h2>
        </div>
        """
    )

    b1.click(Predict, inputs=[marca, modelo, transmision, tipo_de_carroceria, tipo_de_combustible, año, motor, km, km_check],
             outputs=[gr.Label(label="", elem_classes="label-predict")])
    gr.Markdown(f"{n_estimation}")
    gr.Markdown("---")


    gr.Markdown(#"## Simular crédito"
    """
        <div style="text-align: left; max-width: 8000; margin: 0 auto;">
            <h2 style="font-size: 2.2rem; font-weight: 700; margin-bottom: 1rem; margin-top: 1.5rem; display: contents;">🏦 Simular crédito</h2>
        </div>
        """
    )

    with gr.Row():
        n_meses = gr.Radio(label="¿A cuantos meses quieres pagar el crédito?", value=12, choices = [12, 24, 36, 48, 60, 72, 84, 96])
        cuota_init = gr.Number(label="¿De cuanto será la cuota inicial del vehículo?",value=0,  minimum = 0)
        tasa_EA = gr.Number(label="¿De cuanto es la tasa efectiva anual (E.A)?", value=18, minimum = 0.1, step = 0.1, maximum=100)
    b2 = gr.Button("Calcular crédito", elem_classes="button")
    b2.click(SimulateCredit, inputs=[n_meses, cuota_init, tasa_EA], 
             outputs=[gr.Label(label="La cuota mensual del crédito será de", elem_classes="label-predict")])
    gr.Markdown("---")
    dtm_date = datetime.strptime(date, "%Y-%m-%d")

    gr.Markdown(#"## Simular crédito"
    """
        <div style="text-align: left; max-width: 8000; margin: 0 auto;">
            <h2 style="font-size: 2.2rem; font-weight: 700; margin-bottom: 1rem; margin-up: 2rem; display: contents;">📊 Resumen del mercado en plataformas digitales</h2>
        </div>
        """
    )

    median = f'{int(df["precio"].median()*1000000):,}'.replace(",", ".")
    gr.Markdown(f'### El precio promedio de un vehículo usado es de ${median} millones de pesos.')

    #bar_plot = gr.Plot(label = " ", value=plot_median_price_by_brand)

    #b1.click(plot_median_price_by_brand, inputs=[modelo], outputs=[gr.Plot(label = " ")])

    bar_plot2 = gr.Plot(label = " ", value=plot_frequent_by_brand)

demo.launch()