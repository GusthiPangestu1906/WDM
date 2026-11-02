import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from data_loader import load_dataset

dataset = load_dataset()
data = dataset[dataset["Country"] == "Portugal"].copy()

# Bentuk matriks transaksi
transaksi = data.groupby(["InvoiceNo", "StockCode"])["Qty"].sum()
transaksi = transaksi.unstack().reset_index().fillna(0).set_index("InvoiceNo")
transaksi = (transaksi > 0).astype(int)

# 1. Cari frequent itemsets dengan min_support = 0.2
frequent_itemsets = apriori(transaksi.astype(bool), min_support=0.2, use_colnames=True)

# 2. Association rules dengan min_confidence = 0.7
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print("Association Rules:\n", rules[["antecedents", "consequents", "confidence"]])
