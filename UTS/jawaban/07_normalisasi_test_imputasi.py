from sklearn.preprocessing import MinMaxScaler
from data_loader import load_features

data = load_features()
train_data = data.loc[data["Age"].notnull(), ["Sex", "Pclass", "Fare", "Survived"]].copy()
test_data = data.loc[data["Age"].isnull(), ["Sex", "Pclass", "Fare", "Survived"]].copy()

scaler = MinMaxScaler()
scaler.fit(train_data)
test_data_scaled = scaler.transform(test_data)

print("Contoh 5 baris test_data ternormalisasi:")
print(test_data_scaled[:5])
