import pandas as pd

data = pd.read_csv("insurance.csv")
data.info()
data.describe()
print(data.isnull().sum())

