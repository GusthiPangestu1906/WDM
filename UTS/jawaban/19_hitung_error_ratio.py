import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from data_loader import load_features, load_test_dataset, load_test_label

data = load_features()
mask_not_null = data["Age"].notnull()
mask_null = data["Age"].isnull()

scaler_imp = MinMaxScaler()
train_imp = scaler_imp.fit_transform(data.loc[mask_not_null, ["Sex", "Pclass", "Fare", "Survived"]])
test_imp = scaler_imp.transform(data.loc[mask_null, ["Sex", "Pclass", "Fare", "Survived"]])
knn_imp = KNeighborsClassifier(n_neighbors=3).fit(train_imp, data.loc[mask_not_null, "Age"].astype(int))
data.loc[mask_null, "Age"] = knn_imp.predict(test_imp)

train_data = data[["Sex", "Age", "Pclass", "Fare"]].copy()
train_label = data["Survived"].astype(int)

test_dataset = load_test_dataset()
test_label_df = load_test_label()

test_data = test_dataset[["Sex", "Age", "Pclass", "Fare"]].copy()
test_data["Sex"] = test_data["Sex"].map({"female": 0, "male": 1})
valid_idx = test_data.dropna().index
clean_test_data = test_data.loc[valid_idx]
test_label = test_label_df.loc[valid_idx, "Survived"].astype(int).values

scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(clean_test_data)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_data_scaled, train_label)
class_result = knn.predict(test_data_scaled)

error = int(np.sum(class_result != test_label))
total_test = len(clean_test_data)
error_ratio = (error / total_test) * 100
akurasi = 100.0 - error_ratio

print("Total Sampel Data Uji :", total_test)
print("Jumlah Prediksi Salah :", error)
print("Jumlah Prediksi Benar :", total_test - error)
print(f"Error Ratio (%)       : {error_ratio:.2f}%")
print(f"Akurasi (%)           : {akurasi:.2f}%")
