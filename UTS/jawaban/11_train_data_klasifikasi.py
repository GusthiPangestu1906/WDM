from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from data_loader import load_features

data = load_features()
mask_not_null = data["Age"].notnull()
mask_null = data["Age"].isnull()

scaler = MinMaxScaler()
train_imp = scaler.fit_transform(data.loc[mask_not_null, ["Sex", "Pclass", "Fare", "Survived"]])
test_imp = scaler.transform(data.loc[mask_null, ["Sex", "Pclass", "Fare", "Survived"]])
knn = KNeighborsClassifier(n_neighbors=3).fit(train_imp, data.loc[mask_not_null, "Age"].astype(int))
data.loc[mask_null, "Age"] = knn.predict(test_imp)

train_data = data[["Sex", "Age", "Pclass", "Fare"]].copy()

print("train_data untuk klasifikasi:")
print(train_data.head())
print("Dimensi train_data:", train_data.shape)
