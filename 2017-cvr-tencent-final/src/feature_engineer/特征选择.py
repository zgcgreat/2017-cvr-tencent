import pandas as pd
import numpy as np

df = pd.read_csv('../../data/train.csv')
per_positionID = df.groupby('positionID').apply(lambda df: np.mean(df["label"])).reset_index(name='per_positionID')
per_age = df.groupby('age').apply(lambda df: np.mean(df["label"])).reset_index(name='per_age')
print(per_age)