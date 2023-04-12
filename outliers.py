import numpy as np
import pandas as pd

def outliers(data):
    mean = np.mean(data)  # calculate mean
    standardDeviation = np.std(data) # calculate standard deviation
    outliersList = [] # list to allow us to see the pieces of data which will be removed
    for temp in data:
        value = (temp - mean) / standardDeviation # calculates the z score for each value in a column
        value = np.abs(value) # makes it positive
        if value >= 3: # we use 3 as the threshold for the z scores as it is a good limiter to find outliers
            outliersList.append(temp) # adds outlier to list
    return outliersList

dataSet = r"C:\Users\zach6\PycharmProjects\AIMethods\dataSet.xlsx" # opens dataset to find outliers
df = pd.read_excel(dataSet) # reads the excel file into a dataframe

# Loop through columns 1 to 5 and remove outliers
for i in range(1, 6):
    column = df.iloc[1:, i].to_list()
    outlierlist = outliers(column)
    for temp in outlierlist:
        df = df[df.iloc[:, i] != temp]

file = 'test.xlsx'

df.to_excel(file, index=False)