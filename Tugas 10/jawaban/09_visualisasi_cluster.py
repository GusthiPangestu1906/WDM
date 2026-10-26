import os
import numpy as np
import matplotlib.pyplot as plt
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

cluster_order = sorted_centroid.index.tolist()
label_mapping = {
    cluster_order[0]: "Rendah",
    cluster_order[1]: "Sedang",
    cluster_order[2]: "Tinggi",
}
transaksi["Kategori"] = transaksi["Cluster"].map(label_mapping)

df_plot = transaksi.sort_values("Rata_Qty_Per_Transaksi").reset_index(drop=True)

color_map = {
    "Rendah": "#2b83ba",
    "Sedang": "#fdae61",
    "Tinggi": "#d7191c",
}

plt.figure(figsize=(12, 6))
for kat, warna in color_map.items():
    sub = df_plot[df_plot["Kategori"] == kat]
    plt.scatter(
        sub.index,
        sub["Rata_Qty_Per_Transaksi"],
        color=warna,
        label=f"Transaksi {kat}",
        s=70,
        edgecolors="k",
        alpha=0.85,
    )

plt.xticks(
    ticks=range(len(df_plot)),
    labels=df_plot["Country"],
    rotation=90,
    fontsize=8,
)
plt.xlabel("Urutan Country")
plt.ylabel("Rata-rata Qty per Transaksi")
plt.title("Visualisasi Clustering K-Means Rata-rata Qty per Transaksi (k=3, SSE Terbaik)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "hasil_clustering_m10.png")
plt.savefig(output_path, dpi=300)
print(f"Grafik visualisasi disimpan di: {output_path}")

plt.show()
