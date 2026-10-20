from data_loader import load_features

data = load_features()
train_label = data["Survived"].astype(int)

print("train_label untuk klasifikasi (Survived):")
print(train_label.head(10))
print("\nSebaran kelas:")
print(train_label.value_counts())
