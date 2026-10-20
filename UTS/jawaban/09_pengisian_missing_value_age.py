from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from data_loader import load_features

data = load_features()
mask_not_null = data["Age"].notnull()
mask_null = data["Age"].isnull()

train_data = data.loc[mask_not_null, ["Sex", "Pclass", "Fare", "Survived"]].copy()
train_label = data.loc[mask_not_null, "Age"].astype(int)
test_data = data.loc[mask_null, ["Sex", "Pclass", "Fare", "Survived"]].copy()

scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_data_scaled, train_label)
class_result = knn.predict(test_data_scaled)

data.loc[mask_null, "Age"] = class_result

print("Missing values Age setelah pengisian:", data["Age"].isnull().sum())
print("\nSampel 10 baris data setelah Age terisi lengkap:")
print(data.head(10))
