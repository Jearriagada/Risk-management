import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from scipy.stats import norm
from numpy.linalg import cholesky

#Importar activos del portfolio

tickers = ['TSLA', 'AAPL', 'FB', 'NVDA', 'MSFT', 'AMZN']   #tickers de activos
periodo = (dt.datetime.now() - relativedelta(years=5))     #periodo de simulación
df = yf.download(tickers,periodo)['Adj Close']             #creación de df con api de yf

#Definición de w y r

w = np.random.rand(len(tickers))              # w aleatorios
w /= np.sum(w)                                # hacer la suma de los pesos = 1
port_ret = np.log(df/df.shift(1)).dropna()    # retornos logaritmicos 

#Métricas 
ret = port_ret.mean().values #retorno promedio de cada activo
mean = np.dot(w.T,ret)       #retorno promedio del portafolio ajustado por el peso
cov = port_ret.cov()         #matriz varianza covarianza 
desv = np.sqrt(np.dot(w.T,np.dot(cov,w)))    #desviación estándar del portafolio
lp = np.dot(df.iloc[-1].values,w.T)          #último valor del portafolio ajustado por pesos

#VaR histórico

aux = np.dot(port_ret.values,w)                 #Matriz de retornos totales del portafolio 
VaR_hist = np.round(
                ((np.percentile(aux,1))*100),4) #El peor retorno al 99% de confianza redondeado a 4 decimales

print(f"El VaR histórico diario para el portafolio es de {VaR_hist}%") 

#VaR paramétrico 

VaR_Par = mean - (norm.ppf(0.99) * desv * np.sqrt(1)) #la raiz cuadrada indica el periodo en días para el VaR
VaR_Par = np.round(VaR_Par*100,4)

print(f"El VaR paramétrico diario para el portafolio es de {VaR_Par}%") 

#Montecarlo 

simulations = 100000 #caminos aleatorios 
T = 500 #shocks por cada camino aleatorio 
mean_mtx = np.full(shape=(T, len(w)), fill_value=ret/T).T 
portf_simulations = np.full(shape=(T, simulations),fill_value=0.0) #shocks aleatorios 
initial_portf = lp

for i in range(0, simulations):
    Z = np.random.normal(size=(T, len(w)))
    L = cholesky(cov/T)
    daily_ret = mean_mtx + np.inner(L,Z)
    portf_simulations[:,i] = np.cumprod(np.inner(w,daily_ret.T)+1)*initial_portf

# plt.figure(figsize=(12,5))
# plt.style.use('seaborn')
# plt.plot(portf_simulations)
# plt.ylabel('Portfolio value')
# plt.xlabel(
#         'Random paths: ' + str(simulations)+\
#         ' | Shocks: ' + str(T) +\
#         ' | Initial price: ' + str(np.round(lp,4))
#         )
# plt.title('Simulation of stock portfolio')
# plt.show()

perct = np.percentile(portf_simulations,1) # 99% de confianza 
VaR_MC = perct/lp -1
VaR_MC = np.round(VaR_MC*100,4)

print(f"El VaR diario sinulado con montecarlo para el portafolio es de {VaR_MC}%") 
