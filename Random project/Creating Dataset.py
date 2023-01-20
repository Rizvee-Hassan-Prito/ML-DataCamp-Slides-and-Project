# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 17:50:11 2023

@author: User
"""

import pandas as pd

data = pd.read_csv("data.csv")

print(data.head())
print(data.info())

#%%


data=data['M1\tM2\tM3\tM4\tM5\tM6\tM7\tM8\tM9\tN1\tN2\tN3\tN4\tN5\tN6\tN7\tN8\tN9\tP1\tP2\tP3\tP4\tP5\tP6\tP7\tP8\tP9\tcountry\tsource'].str.split('\t', expand=True)
print(data.head())

txt = "M1\tM2\tM3\tM4\tM5\tM6\tM7\tM8\tM9\tN1\tN2\tN3\tN4\tN5\tN6\tN7\tN8\tN9\tP1\tP2\tP3\tP4\tP5\tP6\tP7\tP8\tP9\tcountry\tsource"
x = txt.split("\t")
print(x)


#%%

cols = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'country', 'source']
data.columns = cols

print(data.head())

data.to_csv('SD3_data.csv', index=False)

