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
transaksi["Cluster"] = best_model.labels_

centroid = transaksi.groupby("Cluster")["Rata_Qty_Per_Transaksi"].mean()
sorted_centroid = centroid.sort_values(ascending=True)

cluster_order = sorted_centroid.index.tolist()
label_mapping = {
    cluster_order[0]: "Rendah",
    cluster_order[1]: "Sedang",
    cluster_order[2]: "Tinggi",
}

transaksi["Kategori"] = transaksi["Cluster"].map(label_mapping)

for kat in ["Rendah", "Sedang", "Tinggi"]:
    sub = transaksi[transaksi["Kategori"] == kat]
    print(f"\nKategori Rata-rata Qty {kat} ({len(sub)} negara):")
    for _, row in sub.iterrows():
        print(f"  - {row['Country']}: {row['Rata_Qty_Per_Transaksi']:.2f} unit/transaksi")
