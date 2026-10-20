from data_loader import load_features

data = load_features()

test_data = data.loc[data["Age"].isnull(), ["Sex", "Pclass", "Fare", "Survived"]].copy()

print("test_data untuk imputasi (Age == null):")
print(test_data.head())
print("Jumlah baris missing:", len(test_data))
