import pandas as pd

df = pd.read_csv('../../data/train-p.csv', nrows=10)
print(df.columns)
print(df.head())
