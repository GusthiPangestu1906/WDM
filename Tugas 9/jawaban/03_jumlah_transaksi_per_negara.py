from data_loader import load_dataset

dataset = load_dataset()

transaksi = (
    dataset.groupby("Country")["InvoiceNo"]
    .nunique()
    .reset_index(name="Transaksi")
)

print("Banyaknya transaksi pada tiap negara:")
print(transaksi.to_string(index=False))
