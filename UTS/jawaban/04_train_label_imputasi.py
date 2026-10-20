from data_loader import load_features

data = load_features()

train_label = data.loc[data["Age"].notnull(), "Age"].astype(int)

print("train_label untuk imputasi (Age):")
print(train_label.head(10))
print("Jumlah label:", len(train_label))
