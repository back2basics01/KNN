#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from sklearn.neighbors import NearestNeighbors

melbourne = pd.read_csv("Melbourne_housing_full.csv") 



print(melbourne.head(3))
print(melbourne.columns)



melbourne.dropna(axis = 0, how = 'any', thresh = None, subset = None, inplace = True)  # axis 0 drops rows with missing values



data_set = melbourne.loc[:, ['Price', 'Distance', 'Bedroom2', 'Bathroom', 
                             'Landsize', 'YearBuilt' ]].values



ideal_home = [800000,5,2,3,150,2004]



model = NearestNeighbors(n_neighbors = 3).fit(data_set)



ideal_homes = (model.kneighbors([ideal_home]))



print(ideal_homes)



home_1 = ideal_homes[1][0][0]
home_2 = ideal_homes[1][0][1]
home_3 = ideal_homes[1][0][2]



melbourne.iloc[home_1]


melbourne.iloc[home_2]


melbourne.iloc[home_3]
