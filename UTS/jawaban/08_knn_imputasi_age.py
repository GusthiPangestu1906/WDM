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

print("Hasil klasifikasi 3-NN (prediksi Age):")
print(class_result[:20])
print("\nStatistik prediksi:")
print("Min age:", class_result.min(), "| Max age:", class_result.max(), "| Rata-rata:", round(class_result.mean(), 2))
