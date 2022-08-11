# %% [markdown]
# # Librerias

# %%
import pandas as pd
import datetime as dt
import requests
import json
import numpy as np
from scipy import stats

# %%
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# %% [markdown]
# # Funciones

# %%
def errorback(message):
    """
    Mensaje de error.

    Parametros
    ----------
    :param message: Excepcion

    Return
    ----------
    :return: Mensaje de error por excepcion
    """
    return str(message).split()

# %%
def json_df(url, header):
    """
    Funcion que recibe una API(url junto al header/token)
    y devuelve un DataFrame.

    Parametros
    ----------
    :param url: API url
    :param header: API token

    Return
    ----------
    :return: pandas DataFrame
    """
    try:
        response_API = requests.get(url, headers=header)
        data = response_API.text
        parse_json = json.loads(data)
        return pd.DataFrame(parse_json)
    except AttributeError as e:
        print(f'Expected a Dict instead of {errorback(e)[0]}')
    except Exception as e:
        return e

# %%
def insert_df(data1, data2, col, on):
    """
    Funcion que concatena una columna indice 
    del dataframe principal junto a una serie
    y despues los agrega al principal.
    
    Ejemplos
    --------
    Agregar datos por una columna base del dataframe principal.
    >>> d = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    >>> a = d['col2'].squeeze() #Serie
    >>> df = insert_df(d, a, 'col1', 'col1')
    df
        col1  col2_x  col2_y
    0     1     3     3
    1     2     4     4

    Parametros
    ----------
    :param data1: Dataframe principal
    :param data2: Series
    :param col: Columna del dataframe a concatenar
    :param on: Columna donde se juntaran los datos

    Return
    ---------
    :return: Dataframe 
    """
    try:
        df = pd.concat([data1[f'{col}'], data2], axis=1)
        return data1.merge(df, on=on)
    except Exception as e:
        return e

# %%
def calc_vol(data, fro_m):
    """
    Funcion encargada de calcular la volatilidad teniendo en cuenta la varianza.

    *Sujeto a ajustes*
    
    Parametros
    ----------
    :param data: Dataframe
    :param fro_m: Columna del dataframe a calcular

    Return
    ----------
    :return: Series contenedora de la volatilidad por fila
    """
    try:
        data['PCT %'] = data[f'{fro_m}'].pct_change()
        data_pct = data['PCT %'].iloc[1:]
        vol = data_pct.rolling(20).var(ddof=1)
        vol.name = 'Vol'
        return vol
    except Exception as e:
        return e

# %%
def calc_week(data, on):
    """
    Funcion encargada de asignar las semanas del anio a un Dataframe.

    Parametros
    ----------
    :param data: Dataframe
    :param on: Columna del dataframe para asignar semanas

    Return
    ----------
    :return: Series que contiene los numeros de las semanas del anio
    segun la fecha del Dataframe

    """
    try:
        weeks = data[f'{on}'].dt.isocalendar()
        weeks.drop(columns={'year', 'day'}, inplace=True)
        weeks = insert_df(data, weeks, on, on)
        return weeks
    except Exception as e:
        return e

# %%
def re_idx_name(data, ogcol, rename=False):
    """
    Funcion que re-organiza las columnas en un Dataframe

    Parametros
    ----------
    :param data: Dataframe
    :param ogcol: Orden de las columnas con sus nombres originales
    :param rename: Si se quiere renombrar alguna columna,
    se le pasa un Dict que contenga el nombre de la columna original
    y su nuevo nombre. De la forma {'nombre': 'Nombre'}

    Return
    ----------
    :return: Dataframe reindexado
    """
    try:
        data = data.reindex(ogcol, axis=1)
        if rename:
            data.rename(columns=rename, inplace=True)
            return data
        return data
    except TypeError:
        print('TypeError: Insert a dict in rename place')
    except Exception as e:
        return e

# %%
def clean_reg_data(data, i_date, f_date, on, drop_cols=False):
    """
    Funcion que recibe un dataframe y lo devuelve listo para realizar
    una regresion lineal.

    Parametros
    ----------
    :param data: Dataframe
    :param i_date: Fecha inicial de filtro
    :param f_date: Fecha final de filtro
    :param on: Columna a convertir a numerico *(Recibe datetime)*
    :param drop_cols: Columnas a eliminar del dataframe *(Opcional)*

    Return
    ----------
    :return: Dataframe procesado, columna |datetime| 
    convertida a int
    """
    try:
        date_int = data.loc[(data[f'{on}'] > i_date) & (data[f'{on}'] < f_date)]
        reg_data = date_int.drop(columns=drop_cols)
        reg_data[f'{on}'] = reg_data[f'{on}'].map(dt.datetime.toordinal)
        return reg_data
    except TypeError:
        print('TypeError: Insert a dict in name place')
    except Exception as e:
        return e

# %%
def reg_prep_data(data1, data2):
    """
    Funcion que recibe columnas de datos para entrenar X & Y.

    Parametros
    ----------
    :param data1: Valores de X
    :param data2: Valores de Y

    Return
    ----------
    :return: Tupla contenedora de np.array
    Shape 2
    """
    try:
        x = data1.values
        y = data2.values
        x = x.reshape(-1, 1)
        return x, y
    except Exception as e:
        return e

# %%
def pred_usd(fecha, mod):
    """
    Calculadora de prediccion: USD.
    *Sujeto a ajustes*

    Parametros
    ----------
    :param fecha: Lista contenedora de fecha [aaaa-mm-dd]
    :param mod: model.LinearRegression despues del Fit. 

    Return
    ----------
    :return: Valor del dolar Blue en pesos argentinos
    """
    try:
        pred_date = dt.datetime(fecha[0], fecha[1], fecha[2])
        pred_dateord = pred_date.toordinal()
        print('$', round(mod.predict([[pred_dateord]])[0], 2))
    except Exception as e:
        return e

# %% [markdown]
# # Extraccion datos API, y Procesamiento

# %%
url_blue = 'https://api.estadisticasbcra.com/usd'
url_of = 'https://api.estadisticasbcra.com/usd_of'
url_ev = 'https://api.estadisticasbcra.com/milestones'
header = {'Authorization': 'BEARER eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE2OTEwMDk2OTcsInR5cGUiOiJleHRlcm5hbCIsInVzZXIiOiJjYWphczI1QGhvdG1haWwuY29tIn0.czUNMHIMlah24yk2VPdY9DSppu9zU1GjH4nMndNiDTjkLcRTtwhnXC5A7fbhKH54V2EWSs448RUZEEqkXvzgwg'}

# %%
blue = json_df(url_blue, header)
oficial = json_df(url_of, header)
eventos = json_df(url_ev, header)

# %%
blue.head()

# %%
oficial.head()

# %%
eventos.head()

# %% [markdown]
# Datos 365 dias y Generales

# %%
blue['d'] = pd.to_datetime(blue['d'])
oficial['d'] = pd.to_datetime(oficial['d'])

# %%
blue.rename(columns={'d': 'Date', 'v': 'Blue'}, inplace=True)
oficial.rename(columns={'d': 'Date', 'v': 'Oficial'}, inplace=True)

# %%
blue_22 = blue.iloc[-366:]
of_22 = oficial.iloc[-366:]

# %%
blue_22.columns = ['Date', 'Blue']
of_22.columns = ['Date', 'Oficial']

# %%
blue_22.set_index('Date')
of_22.set_index('Date')
blue.set_index('Date')
oficial.set_index('Date')

# %%
valores = blue_22.merge(of_22, on='Date')
valores_g = blue.merge(oficial, on='Date')

# %% [markdown]
# Adicion de Dias

# %%
day_week = {0: 'Lunes', 1: 'Martes', 2: 'Miercoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sabado', 6: 'Domingo'}

# %%
datos = valores
datos['Day'] = datos['Date'].dt.day_of_week
datos['Day'].replace(day_week, inplace=True)
datos

# %%
datos_g = valores_g
datos_g['Day'] = datos_g['Date'].dt.day_of_week
datos_g['Day'].replace(day_week, inplace=True)
datos_g

# %% [markdown]
# Eliminacion de Outliers (Opcional)

# %%
#valores.plot(kind='box')

# %%
#q1 = np.percentile(valores.Blue, 25)
#q3 = np.percentile(valores.Blue, 75)
#quart = stats.iqr(valores.Blue)

# %%
#upper = np.where(valores.Blue >= (q3 + 1.5 * quart))
#lower = np.where(valores.Blue <= (q1 - 1.5 * quart))

# %%
#valores.drop(upper[0], inplace=True)
#valores.drop(lower[0], inplace=True)

# %% [markdown]
# Brechas

# %%
brecha = ((datos['Blue'] / datos['Oficial']) -1) * 100
brecha.name = 'Gap'
datos = insert_df(datos, brecha, 'Date', 'Date')

# %%
brecha_g = ((datos_g['Blue'] / datos_g['Oficial']) -1) * 100
brecha_g.name = 'Gap'
datos_g = insert_df(datos_g, brecha_g, 'Date', 'Date')

# %%
datos_g

# %% [markdown]
# # 1.
# Día con mayor variación en la brecha

# %%
brecha_max = datos['Gap'].idxmax()
datos.iloc[brecha_max]

# %% [markdown]
# Volatilidad

# %%
datos_vol = calc_vol(datos, 'Blue')
datos['Vol'] = datos_vol

# %%
datos_g_vol = calc_vol(datos_g, 'Blue')
datos_g['Vol'] = datos_g_vol

# %%
datos_g

# %% [markdown]
# # 2.
# Top 5 días con mayor volatilidad

# %%
max_vol = datos.iloc[datos['Vol'].nlargest(n=5).index]
max_vol.sort_index()

# %% [markdown]
# Adicion de Semanas

# %%
datos = calc_week(datos, 'Date')
datos_g = calc_week(datos_g, 'Date')

# %% [markdown]
# Organizacion de Columnas

# %%
dat_col = ['Date', 'Blue', 'Oficial', 'Day', 'week', 'Gap', 'PCT %', 'Vol']

# %%
datos = re_idx_name(datos, dat_col, {'week': 'Week'})
datos_g = re_idx_name(datos_g, dat_col, {'week': 'Week'})

# %% [markdown]
# # 3.
# Semana con mayor variación en la brecha

# %%
large_week = datos.iloc[datos['Gap'].nlargest(10).index]
large_week = large_week.Week.unique()
week_gap = [{f'Week {large_week[x]}': round(datos[datos['Week'] == large_week[x]].Gap.sum(), 2)} 
            for x in range(len(large_week))]
week_gap

# %% [markdown]
# # 4.
# Día de la semana donde hay mayor variación en la brecha

# %%
large_day = datos.iloc[datos['Gap'].nlargest(20).index]
large_day = large_day.Day.unique()
day_gap = [{large_day[x]: round(datos[datos['Day'] == large_day[x]].Gap.sum(), 2)} 
            for x in range(len(large_day))]
day_gap

# %%
datos.sort_values('Date', inplace=True)

# %% [markdown]
# Adicion de eventos

# %%
eventos['d'] = pd.to_datetime(eventos['d'])
eventos.rename(columns={'d': 'Date', 'e': 'Evento', 't': 'Position'}, inplace=True)
eventos.head(10)

# %%
even = eventos.drop(columns='Position')

# %%
even.drop_duplicates(subset='Date', inplace=True)
even.reset_index(inplace=True, drop=True)

# %%
datos_g = pd.merge(datos_g, even, how='left')

# %%
datos_g.drop('Evento', inplace=True, axis=1)

# %%
datos_g['Evento'].fillna('No Event', inplace=True)
datos_g

# %% [markdown]
# # 5.
# Con la info histórica del valor del dólar y del blue, realizar un análisis exploratorio. Cruzar la data con sucesos importantes a nivel político-económico y graficar mes a mes

# %%
fig = px.line(datos_g, x="Date", y=['Blue', 'Oficial'], custom_data=[datos_g['Evento']])
hovertemp = '<b>Pesos:</b> %{y} '
hovertemp += '<b>Fecha:</b> %{x} '
hovertemp += '<b>Evento:</b> %{customdata[0]}'
fig.update_layout(title='Historico dolar Blue frente al Oficial', xaxis_title='Año',
                yaxis_title='Pesos Argentinos', hovermode='x unified')
fig.update_traces(hovertemplate=hovertemp)
hovertemp2 = '<b>Brecha:</b> %{y} %'
fig2 = px.line(datos_g, x='Date', y='Gap')
fig2.update_traces(line_color='#34eba8', hovertemplate=hovertemp2)
fig.add_trace(fig2.data[0])
fig.show()

# %% [markdown]
# # 6.
# Implementar una regresión lineal (una para cada tipo de dólar) para predecir el valor del dólar en:
#             * 3 meses
#             * 6 meses
#             * 12 meses

# %%
reg_12 = clean_reg_data(datos_g, datos_g.Date.iloc[-367], datos_g.Date.iloc[-1], 'Date', {'Day', 'Week', 'Gap', 'PCT %', 'Vol', 'Evento'})
reg_12

# %%
model = LinearRegression(fit_intercept=True)

# %% [markdown]
# Entrenamiento para 3, 6 y 12 meses

# %%
x12_b, y12_b = reg_prep_data(reg_12['Date'], reg_12['Blue'])
#x12_o, y12_o = reg_prep_data(reg_12['Date'], reg_12['Oficial'])

# %%
X_train_12b, X_test_12b, y_train_12b, y_test_12b = train_test_split(x12_b, y12_b, test_size=0.7, random_state=33)
#X_train_12o, X_test_12o, y_train_12o, y_test_12o = train_test_split(x12_b, y12_b, test_size=0.7, random_state=33)

# %% [markdown]
# Dolar Blue

# %%
model.fit(X_train_12b, y_train_12b)

# %%
y_train_pred = model.predict(X_train_12b)
y_test_pred = model.predict(X_test_12b)

# %%
print('Error en datos de train:', mean_squared_error(y_train_12b, y_train_pred))
print('Error en datos de test:', mean_squared_error(y_test_12b, y_test_pred))

# %%
plt.figure(figsize = (7,6))
plt.scatter(X_train_12b, y_train_12b,  color='green', label = 'Blue Train')
plt.plot(X_train_12b, y_train_pred, color='k', linestyle = '--', label = 'Prediccion Train')

plt.scatter(X_test_12b, y_test_12b,  color='blue', label = 'Blue Test')
plt.plot(X_test_12b, y_test_pred, color='red', linewidth=3.0, label = 'Prediccion Test')

plt.legend()
plt.show()

# %% [markdown]
# # Predicciones desde (2022-08-03)

# %% [markdown]
# Prediccion 3 meses

# %%
dia = 3
mes = 11
anio = 2022
pred_date = dt.datetime(anio, mes, dia)
pred_dateord = pred_date.toordinal()
print('$', round(model.predict([[pred_dateord]])[0], 2))

# %% [markdown]
# Prediccion 6 meses

# %%
dia = 3
mes = 2
anio = 2023
pred_date = dt.datetime(anio, mes, dia)
pred_dateord = pred_date.toordinal()
print('$', round(model.predict([[pred_dateord]])[0], 2))

# %% [markdown]
# Prediccion 12 meses

# %%
dia = 3
mes = 8
anio = 2023
pred_date = dt.datetime(anio, mes, dia)
pred_dateord = pred_date.toordinal()
print('$', round(model.predict([[pred_dateord]])[0], 2))

# %% [markdown]
# # 7.
# Bonus opcional: Realizar una calculadora de predicción de aumento del dólar

# %%
fecha = [int(x) for x in input('Ingrese año, mes y dia:').split()]
pred_usd(fecha, model)

# %% [markdown]
# # 8.
# 
# Últimos 4 años:
# Mejor momento para comprar dolár oficial y venderlo a dolár blue

# %%
idx_4 = datos_g.loc[datos_g.Date == '2018-08-03'].index
idx_4[0] - datos_g.last_valid_index()

# %%
last_4 =  datos_g.loc[(datos_g['Date'] > datos_g.Date.iloc[-969]) & (datos_g['Date'] < datos_g.Date.iloc[-1])]

# %%
last_4.loc[last_4.Gap.idxmax()]
