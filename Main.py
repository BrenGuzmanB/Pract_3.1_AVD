"""
Created on Mon Nov 13 14:24:26 2023

@author: Bren Guzmán, María José Merino, Brenda García
"""

#%% LIBRERÍAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% CARGAR ARCHIVO

df = pd.read_csv("Aquifer_Petrignano.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')


#%% 1. Visualización de los datos. 
#Representaciones gráficas donde se observe el comportamiento de cada característica a lo largo del tiempo.

# Configuración del estilo de las gráficas
sns.set(style="whitegrid")

# Configuración del tamaño de la figura
plt.figure(figsize=(14, 8))

# Características
features = ['Rainfall_Bastia_Umbra', 'Temperature_Bastia_Umbra', 
            'Temperature_Petrignano', 'Volume_C10_Petrignano', 
            'Hydrometry_Fiume_Chiascio_Petrignano']

# Objetivos
targets = ['Depth_to_Groundwater_P24', 'Depth_to_Groundwater_P25']


for feature in features + targets:
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df[feature])
    plt.title(f'Comportamiento de {df[feature].name} a lo largo del Tiempo')
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    plt.show()


#%% 2. Asegurarse que los datos tienen una distancia en el tiempo equidistante y ordenada de forma cronológica.

# Ordenar el DataFrame por la columna de fechas
df = df.sort_values(by='Date')

# Calcular las diferencias de tiempo entre fechas
time_diff = df['Date'].diff()

# Verificar si todas las diferencias son iguales
equidistant = time_diff.iloc[1:].eq(time_diff.iloc[1]).all()

if equidistant:
    print("Los datos tienen una distancia de tiempo equidistante.")
else:
    print("Los datos no tienen una distancia de tiempo equidistante.")
