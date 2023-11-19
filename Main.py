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

#%% 3. Manejo de valores faltantes.

#%%% Modificar columnas
# En las gráficas se ve que los ceros en estas columnas en realidad son valores faltantes.

columns_to_change = ['Temperature_Petrignano', 'Hydrometry_Fiume_Chiascio_Petrignano']

# Así que hay que reemplazar los 0 por NaN en las columnas seleccionadas

df[columns_to_change] = df[columns_to_change].replace(0, np.nan)

#%%% Gráficas con valores faltantes

# Establecer la columna 'Date' como el índice
df.set_index('Date', inplace=True, drop=False)

features = ['Rainfall_Bastia_Umbra', 'Temperature_Bastia_Umbra', 'Temperature_Petrignano', 'Volume_C10_Petrignano', 'Hydrometry_Fiume_Chiascio_Petrignano']
targets = ['Depth_to_Groundwater_P24', 'Depth_to_Groundwater_P25']

fig, axes = plt.subplots(nrows=(len(features + targets)), ncols=1, figsize=(10, 5 * len(features + targets)))

for i, column in enumerate(features + targets):
    axes[i].plot(df.index, df[column], color='blue')
    
    # Marcar la zona de valores faltantes con líneas verticales rojas
    missing_values = df[df[column].isnull()].index
    for value in missing_values:
        axes[i].axvline(x=value, color='red', alpha=0.2, linestyle='-', linewidth=2)

    axes[i].set_title(f'Valores Faltantes en {column}')
    axes[i].set_xlabel('Fecha')
    axes[i].set_ylabel('Valor')

plt.tight_layout()
plt.show()

#%%% Imputación

from sklearn.impute import SimpleImputer

columns_to_impute = ['Rainfall_Bastia_Umbra', 'Temperature_Bastia_Umbra', 'Temperature_Petrignano',
                      'Volume_C10_Petrignano', 'Hydrometry_Fiume_Chiascio_Petrignano',
                      'Depth_to_Groundwater_P24', 'Depth_to_Groundwater_P25']

# Definir la ventana de tiempo para imputación local (+/- 30 días)
window_size = 30

# Iterar sobre cada columna
for column in columns_to_impute:
    
    # Imputación con la mediana local
    median_imputer = SimpleImputer(strategy='median')
    
    # Imputación con la media local
    mean_imputer = SimpleImputer(strategy='mean')

    # Nueva columna para la mediana local
    df[f'{column}_imputed_median_local'] = df[column]
    # Nueva columna para la media local
    df[f'{column}_imputed_mean_local'] = df[column]

    for idx in df[df[column].isnull()].index:
        # Seleccionar la ventana de tiempo alrededor del valor faltante
        local_data = df[column][idx - pd.Timedelta(days=window_size):idx + pd.Timedelta(days=window_size)]

        # Verificar si hay valores nulos o fuera de los límites de fechas en la ventana local
        if local_data.isnull().sum() > 30 or local_data.index.min() < df.index.min() or local_data.index.max() > df.index.max():
            # Imputar el valor faltante con la mediana general
            df.at[idx, f'{column}_imputed_median_local'] = median_imputer.fit_transform(df[[column]])[0, 0]

            # Imputar el valor faltante con la media general
            df.at[idx, f'{column}_imputed_mean_local'] = mean_imputer.fit_transform(df[[column]])[0, 0]
        else:
            # Imputar el valor faltante con la mediana local
            df.at[idx, f'{column}_imputed_median_local'] = median_imputer.fit_transform(local_data.values.reshape(-1, 1))[0, 0]

            # Imputar el valor faltante con la media local
            df.at[idx, f'{column}_imputed_mean_local'] = mean_imputer.fit_transform(local_data.values.reshape(-1, 1))[0, 0]

#%%% Gráficas después del manejo de valores faltantes

# Lista de columnas imputadas con la media y la mediana
columns_imputed_median = [f'{col}_imputed_median_local' for col in columns_to_impute]
columns_imputed_mean = [f'{col}_imputed_mean_local' for col in columns_to_impute]

# Grid
fig, axes = plt.subplots(nrows=len(columns_imputed_mean), ncols=1, figsize=(10, 5 * len(columns_imputed_mean)))


for i, column in enumerate(columns_imputed_mean):
    # Máscara para identificar valores imputados
    imputed_mask = df[columns_to_impute[i]].isnull()

    # Graficar la columna imputada con la media
    axes[i].plot(df.index, df[column], label='Datos Originales', color='blue')
    axes[i].scatter(df.index[imputed_mask], df[column][imputed_mask], color='green', label='Imputado (Media)', marker='o')

    axes[i].set_title(f'Imputación con Media - {column}')
    axes[i].set_xlabel('Fecha')
    axes[i].set_ylabel('Valor')
    axes[i].legend()

plt.tight_layout()
plt.show()

# Grid
fig, axes = plt.subplots(nrows=len(columns_imputed_median), ncols=1, figsize=(10, 5 * len(columns_imputed_median)))


for i, column in enumerate(columns_imputed_median):
    # Máscara para identificar valores imputados
    imputed_mask = df[columns_to_impute[i]].isnull()

    # Graficar la columna imputada con la mediana
    axes[i].plot(df.index, df[column], label='Datos Originales', color='blue')
    axes[i].scatter(df.index[imputed_mask], df[column][imputed_mask], color='green', label='Imputado (Mediana)', marker='o')

    axes[i].set_title(f'Imputación con Mediana - {column}')
    axes[i].set_xlabel('Fecha')
    axes[i].set_ylabel('Valor')
    axes[i].legend()

plt.tight_layout()
plt.show()

