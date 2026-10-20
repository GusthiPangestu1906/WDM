from data_loader import load_test_dataset

test_dataset = load_test_dataset()
test_data = test_dataset[["Sex", "Age", "Pclass", "Fare"]].copy()
test_data["Sex"] = test_data["Sex"].map({"female": 0, "male": 1})
clean_test_data = test_data.dropna()

print("test_data setelah missing values dihilangkan:")
print(clean_test_data.head())
print("Jumlah sampel bersih:", len(clean_test_data))
print("Jumlah baris yang dibuang:", len(test_data) - len(clean_test_data))
