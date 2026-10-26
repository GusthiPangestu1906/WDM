from data_loader import load_dataset

dataset = load_dataset()

invoice_qty = (
    dataset.groupby(["Country", "InvoiceNo"])["Qty"].sum().reset_index()
)

transaksi = (
    invoice_qty.groupby("Country")["Qty"]
    .mean()
    .reset_index(name="Rata_Qty_Per_Transaksi")
)

print("Rata-rata jumlah barang (Qty) per transaksi tiap negara:")
print(transaksi.to_string(index=False))
