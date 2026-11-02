import pandas as pd
from data_loader import load_dataset

dataset = load_dataset()
data = dataset[dataset["Country"] == "Portugal"].copy()

# 1 kode InvoiceNo = 1 transaksi, kolom berisi StockCode
transaksi = data.groupby(["InvoiceNo", "StockCode"])["Qty"].sum()
transaksi = transaksi.unstack().reset_index().fillna(0).set_index("InvoiceNo")
transaksi = (transaksi > 0).astype(int)

print("Tabel Transaksi:\n", transaksi)
