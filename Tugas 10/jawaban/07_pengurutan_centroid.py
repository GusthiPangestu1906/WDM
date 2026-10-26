import numpy as np
from sklearn.cluster import KMeans
from data_loader import load_dataset

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
transaksi["Cluster"] = best_model.labels_

centroid = transaksi.groupby("Cluster")["Rata_Qty_Per_Transaksi"].mean()
sorted_centroid = centroid.sort_values(ascending=True)

print("Pengurutan posisi centroid secara ascending:")
for rank, (c_id, val) in enumerate(sorted_centroid.items()):
    print(f"Urutan {rank} -> Cluster ID {c_id}: centroid = {val:.2f}")
