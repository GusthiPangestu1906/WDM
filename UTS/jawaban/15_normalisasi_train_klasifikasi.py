import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from data_loader import load_features

data = load_features()
mask_not_null = data["Age"].notnull()
mask_null = data["Age"].isnull()

scaler_imp = MinMaxScaler()
train_imp = scaler_imp.fit_transform(data.loc[mask_not_null, ["Sex", "Pclass", "Fare", "Survived"]])
test_imp = scaler_imp.transform(data.loc[mask_null, ["Sex", "Pclass", "Fare", "Survived"]])
knn_imp = KNeighborsClassifier(n_neighbors=3).fit(train_imp, data.loc[mask_not_null, "Age"].astype(int))
data.loc[mask_null, "Age"] = knn_imp.predict(test_imp)

train_data = data[["Sex", "Age", "Pclass", "Fare"]].copy()

scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)

print("Nilai Min setiap atribut:")
print(pd.Series(scaler.data_min_, index=train_data.columns))
print("\nNilai Max setiap atribut:")
print(pd.Series(scaler.data_max_, index=train_data.columns))
print("\nContoh 5 baris train_data ternormalisasi:")
print(train_data_scaled[:5])
