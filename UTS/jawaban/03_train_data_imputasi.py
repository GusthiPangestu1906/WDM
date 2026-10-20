from data_loader import load_features

data = load_features()

train_data = data.loc[data["Age"].notnull(), ["Sex", "Pclass", "Fare", "Survived"]].copy()

print("train_data untuk imputasi (Age != null):")
print(train_data.head())
print("Jumlah baris train_data:", len(train_data))
