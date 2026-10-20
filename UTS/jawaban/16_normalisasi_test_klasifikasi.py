from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from data_loader import load_features, load_test_dataset

data = load_features()
mask_not_null = data["Age"].notnull()
mask_null = data["Age"].isnull()

scaler_imp = MinMaxScaler()
train_imp = scaler_imp.fit_transform(data.loc[mask_not_null, ["Sex", "Pclass", "Fare", "Survived"]])
test_imp = scaler_imp.transform(data.loc[mask_null, ["Sex", "Pclass", "Fare", "Survived"]])
knn_imp = KNeighborsClassifier(n_neighbors=3).fit(train_imp, data.loc[mask_not_null, "Age"].astype(int))
data.loc[mask_null, "Age"] = knn_imp.predict(test_imp)

train_data = data[["Sex", "Age", "Pclass", "Fare"]].copy()

test_dataset = load_test_dataset()
test_data = test_dataset[["Sex", "Age", "Pclass", "Fare"]].copy()
test_data["Sex"] = test_data["Sex"].map({"female": 0, "male": 1})
clean_test_data = test_data.dropna()

scaler = MinMaxScaler()
scaler.fit(train_data)
test_data_scaled = scaler.transform(clean_test_data)

print("Contoh 5 baris test_data ternormalisasi:")
print(test_data_scaled[:5])
