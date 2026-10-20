from data_loader import load_test_dataset, load_test_label

test_dataset = load_test_dataset()
test_label_df = load_test_label()

test_data = test_dataset[["Sex", "Age", "Pclass", "Fare"]].copy()
test_data["Sex"] = test_data["Sex"].map({"female": 0, "male": 1})
valid_idx = test_data.dropna().index

test_label = test_label_df.loc[valid_idx, "Survived"].astype(int).values

print("test_label sesuai urutan baris test_data no. 13:")
print(test_label[:20])
print("Jumlah test_label:", len(test_label))
