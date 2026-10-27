from data_loader import load_dataset
import numpy as np
from sklearn.cluster import KMeans

dataset = load_dataset()
invoice_qty = dataset.groupby(["Country", "InvoiceNo"])["Qty"].sum().reset_index()
transaksi = (
    invoice_qty.groupby("Country")["Qty"]
    .mean()
    .reset_index(name="Rata_Qty_Per_Transaksi")
)

X = transaksi[["Rata_Qty_Per_Transaksi"]].values

cluster_models = []
cluster_val = []

for i in range(10):
    kmeans = KMeans(n_clusters=3, init="random", n_init=1, random_state=i)
    kmeans.fit(X)
    cluster_models.append(kmeans)
    cluster_val.append(kmeans.inertia_)

best_idx = int(np.argmin(cluster_val))
best_model = cluster_models[best_idx]
best_sse = cluster_val[best_idx]

transaksi["Cluster"] = best_model.labels_

print(f"Model terbaik: Iterasi ke-{best_idx + 1} dengan SSE terkecil = {best_sse:.4f}")
print(transaksi[["Country", "Rata_Qty_Per_Transaksi", "Cluster"]].to_string(index=False))
