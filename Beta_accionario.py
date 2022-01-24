import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta as rd
from pandas_datareader import DataReader as pdr
from scipy.stats import linregress

#Obtención de fechas 
start = dt.datetime.now() - rd(years=5)
end = dt.datetime.now()

#Data del mercado
stock0 = input("Escriba el símbolo de la acción ")
mkt0 = input("Escriba el símbolo del índice ")

stock = pdr([stock0], 'yahoo',start,end)['Close']
mkt = pdr([mkt0], 'yahoo',start,end)['Close']

def sync_retornos(s, b):
    """
    Esta función realiza una depuración de las fechas de cada DataFrame,
    dejando solo los retornos en los cuales las fechas coinciden,
    es una depuración para cualquier caso posible.

    """
    #Definición de etiquetas
    col = list(s.columns) + list(b.columns)
    s = s.pct_change().dropna()
    b = b.pct_change().dropna()
    #Obtención de fechas
    s['Date'] = s.index
    s = s.reset_index(drop=True)
    b['Date'] = b.index
    b = b.reset_index(drop=True)
    fechas_s = list(s['Date'].values)
    fechas_b = list(b['Date'].values)
    fechas = list(set(fechas_s) & set(fechas_b))
    #################################
    ###sincronización de los datos###
    #################################
    s_sync = s[s['Date'].isin(fechas)]
    s_sync.sort_values(by = 'Date', ascending = True)
    s_sync = s_sync.reset_index(drop = True)
    b_sync = b[b['Date'].isin(fechas)]
    b_sync.sort_values(by = 'Date', ascending = True)
    b_sync = b.reset_index(drop = True)
    #DataFrame final
    df_retornos = pd.merge(s_sync,b_sync, on='Date')
    df_retornos = df_retornos[['Date'] + col]
    return df_retornos

df_retornos = sync_retornos(stock,mkt)
print(df_retornos)

###Regresión lineal###

y = df_retornos[stock0].values
x = df_retornos[mkt0].values
lin = linregress(x,y)
b = np.round(lin.slope,4)
a = np.round(lin.intercept,4)
r2 = np.round(lin.rvalue**2,4)
pvalue = lin.pvalue
h0 = pvalue > 0.05

print(
    f"\nResumen de la regresion",
    f"\n-----------------------\n\n",
    f"Numero de observaciones: {len(df_retornos)}\n",
    f"R-cuadrado:              {(r2)}\n\n",
    f"Beta de la acción:       {(b)}\n",
    f"Intercepto:              {(a)}\n",
    f"P-Value:                 {(round(pvalue))}\n",
    f"La hipótesis nula es {h0}"
    )

# Gráfica de la regresión
yfinal = a + b * x
plt.figure()
plt.style.use('seaborn')
plt.title('Regresión lineal | Muestra: ' + str(len(x)))
plt.scatter(x,y)
plt.ylabel(stock0)
plt.xlabel(
    mkt0 + '\n' + '\n'\
    + 'Beta accionario: ' + str(b) + '  |  Intercepto: ' + str(a)\
    + ' |  R-cuadrado:  ' + str(r2) + '  |  Hipótesis nula: ' + str(h0)
)
plt.plot(x, yfinal, color = 'red')
plt.show()