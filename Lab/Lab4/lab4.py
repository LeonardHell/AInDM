import pandas as pd
import numpy as np

file_path = "credit_risk_dataset.csv"
df = pd.read_csv(file_path)
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())