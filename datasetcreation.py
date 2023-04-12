import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


df = pd.read_excel('test.xlsx')
cols_to_standardize = ['T', 'W', 'SR', 'DSP', 'DRH', 'PanE']
scaler = MinMaxScaler()
df[cols_to_standardize] = np.round(scaler.fit_transform(df[cols_to_standardize]), 2)
train_data, val_data, test_data = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])


train_data.iloc[:, :7].to_excel('train_data.xlsx', index=False)
val_data.iloc[:, :7].to_excel('val_data.xlsx', index=False)
test_data.iloc[:, :7].to_excel('test_data.xlsx', index=False)