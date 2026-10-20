import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_loader import load_features

data = load_features()
train_data = data.loc[data["Age"].notnull(), ["Sex", "Pclass", "Fare", "Survived"]].copy()

scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)

print("Nilai Min setiap atribut:")
print(pd.Series(scaler.data_min_, index=train_data.columns))
print("\nNilai Max setiap atribut:")
print(pd.Series(scaler.data_max_, index=train_data.columns))
print("\nContoh 5 baris train_data ternormalisasi:")
print(train_data_scaled[:5])
