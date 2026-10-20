from data_loader import load_dataset

dataset = load_dataset()

print("Dataset titanic.csv:")
print(dataset.head())
print("Dimensi:", dataset.shape)
