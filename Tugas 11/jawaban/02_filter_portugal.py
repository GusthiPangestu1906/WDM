from data_loader import load_dataset

dataset = load_dataset()
data = dataset[dataset["Country"] == "Portugal"].copy()

print("Data transaksi negara Portugal:")
print(data.head())
print("\nJumlah baris transaksi Portugal:", len(data))
print("Jumlah InvoiceNo unik di Portugal:", data["InvoiceNo"].nunique())

