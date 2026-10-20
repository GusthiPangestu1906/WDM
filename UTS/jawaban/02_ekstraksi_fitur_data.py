from data_loader import load_dataset

dataset = load_dataset()

data = dataset[["Sex", "Age", "Pclass", "Fare", "Survived"]].copy()
data["Sex"] = data["Sex"].map({"female": 0, "male": 1})

print("Fitur data:")
print(data.head())
print("\nJumlah missing values per kolom:")
print(data.isnull().sum())
