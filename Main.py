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