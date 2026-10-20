import pandas as pd
from data_loader import get_csv_path

test_dataset = pd.read_csv(get_csv_path("titanic_test.csv"))
test_label = pd.read_csv(get_csv_path("titanic_testlabel.csv"))

print("test_dataset:")
print(test_dataset.head())
print("Jumlah sampel data uji:", len(test_dataset))

print("\ntest_label:")
print(test_label.head())
print("Jumlah sampel label uji:", len(test_label))
