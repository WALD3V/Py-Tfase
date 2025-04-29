import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

# ---------- CONFIGURACIÓN ----------
RUTA_CSV = "Libro1.csv"     # Cambia este nombre si tu archivo tiene otro nombre
COLUMNA_FECHA = "FECHA"
COLUMNA_VALOR = "CANTIDAD"
PERIODO_ESTACIONALIDAD = 12   # Cambia según tus datos (12 = mensual con patrón anual)

# ---------- CARGA DE DATOS ----------
# Cargar el archivo CSV, indicando el delimitador y que FECHA sea de tipo datetime
df = pd.read_csv(RUTA_CSV, sep=';', parse_dates=[COLUMNA_FECHA], index_col=COLUMNA_FECHA)

# Ordenamos el DataFrame por la columna 'FECHA' (asegurando que está en orden cronológico)
df.sort_index(inplace=True)

# Usamos la columna 'CANTIDAD' para la serie temporal
# Convertimos la columna CANTIDAD a numérico, forzando a NaN si no es numérico
df[COLUMNA_VALOR] = pd.to_numeric(df[COLUMNA_VALOR], errors='coerce')

# ---------- FUNCIÓN: ESTACIONARIEDAD ----------
def es_estacionaria(serie, alpha=0.05):
    try:
        resultado = adfuller(serie.dropna())
        return resultado[1] < alpha  # p-value
    except Exception as e:
        print(f"Error al comprobar estacionariedad: {e}")
        return False  # Si ocurre un error, asumimos que no es estacionaria

# ---------- FUNCIÓN: ESTACIONALIDAD ----------
def tiene_estacionalidad(serie, periodo=12):
    try:
        descomp = seasonal_decompose(serie, period=periodo, model='additive')
        var_serie = np.var(serie.dropna())
        var_estacional = np.var(descomp.seasonal.dropna())
        return var_estacional / var_serie > 0.1
    except Exception as e:
        print(f"Error al detectar estacionalidad: {e}")
        return False

# ---------- ANÁLISIS ----------
serie = df[COLUMNA_VALOR].copy()

# Si no es estacionaria, diferenciamos
d = 0
if not es_estacionaria(serie):
    serie = serie.diff().dropna()
    d = 1

# Detectamos estacionalidad
usar_sarima = tiene_estacionalidad(serie, periodo=PERIODO_ESTACIONALIDAD)

# ---------- AJUSTE DEL MODELO ----------
try:
    if usar_sarima:
        print("Usando modelo SARIMA")
        modelo = SARIMAX(serie, order=(1, d, 1), seasonal_order=(1, 1, 1, PERIODO_ESTACIONALIDAD))
    else:
        print("Usando modelo ARIMA")
        modelo = ARIMA(serie, order=(1, d, 1))

    resultado = modelo.fit()
except Exception as e:
    print(f"Error al ajustar el modelo: {e}")
    resultado = None  # Si no se puede ajustar el modelo, resultado es None

# ---------- PRONÓSTICO ----------
if resultado is not None:
    try:
        pasos = 12  # Meses a predecir (ajusta según necesites)
        pred = resultado.forecast(steps=pasos)
    except Exception as e:
        print(f"Error al realizar el pronóstico: {e}")
        pred = None  # Si no se puede hacer el pronóstico, pred es None
else:
    pred = None

# ---------- GRAFICAR ----------
if pred is not None:
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(serie, label='Histórico')
        plt.plot(pred, label='Pronóstico', color='red')
        plt.title("Proyección de Ventas")
        plt.xlabel("Fecha")
        plt.ylabel("Ventas")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error al graficar los resultados: {e}")
else:
    print("No se pudo realizar el pronóstico, no se graficarán los resultados.")
