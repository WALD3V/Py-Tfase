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
RUTA_CSV = "data_ventasss.csv"     # Cambia este nombre si tu archivo tiene otro nombre
COLUMNA_FECHA = "DATE"
COLUMNA_VALOR = "CANT"
PERIODO_ESTACIONALIDAD = 12   # Cambia según tus datos (12 = mensual con patrón anual)

# ---------- CARGA DE DATOS ----------
df = pd.read_csv(RUTA_CSV, sep=';', parse_dates=[COLUMNA_FECHA], index_col=COLUMNA_FECHA)
serie_original = df[COLUMNA_VALOR].sort_index()

# ---------- FUNCIÓN: ESTACIONARIEDAD ----------
def es_estacionaria(serie, alpha=0.05):
    resultado = adfuller(serie.dropna())
    return resultado[1] < alpha  # p-value

# ---------- FUNCIÓN: ESTACIONALIDAD ----------
def tiene_estacionalidad(serie, periodo=12):
    try:
        descomp = seasonal_decompose(serie, period=periodo, model='additive')
        var_serie = np.var(serie.dropna())
        var_estacional = np.var(descomp.seasonal.dropna())
        return var_estacional / var_serie > 0.1
    except:
        return False

# ---------- ANÁLISIS ----------
serie = serie_original.copy()

# Si no es estacionaria, diferenciamos
d = 0
if not es_estacionaria(serie):
    serie = serie.diff().dropna()
    d = 1

# Detectamos estacionalidad
usar_sarima = tiene_estacionalidad(serie_original, periodo=PERIODO_ESTACIONALIDAD)

# ---------- AJUSTE DEL MODELO ----------
if usar_sarima:
    print("Usando modelo SARIMA")
    modelo = SARIMAX(serie_original, order=(1, d, 1), seasonal_order=(1, 1, 1, PERIODO_ESTACIONALIDAD))
else:
    print("Usando modelo ARIMA")
    modelo = ARIMA(serie_original, order=(1, d, 1))

resultado = modelo.fit()

# ---------- PRONÓSTICO ----------
pasos = 12  # Meses a predecir (ajusta según necesites)
pred = resultado.forecast(steps=pasos)

# ---------- GRAFICAR ----------
plt.figure(figsize=(10, 5))
plt.plot(serie_original, label='Histórico')
plt.plot(pred, label='Pronóstico', color='red')
plt.title("Proyección de Ventas")
plt.xlabel("Fecha")
plt.ylabel("Ventas")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()