from data_loader import load_dataset

dataset = load_dataset()

print("Dataset transaction.csv:")
print(dataset.head())
print("\nDimensi dataset:", dataset.shape)

