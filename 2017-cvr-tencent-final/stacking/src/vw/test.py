import pandas as pd

data_path = '../../data/'

df = pd.read_csv(data_path+'strain.csv')
print(len(df))
