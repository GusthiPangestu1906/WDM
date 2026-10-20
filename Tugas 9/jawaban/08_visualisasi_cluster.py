import os
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from data_loader import load_dataset

dataset = load_dataset()
transaksi = (
    dataset.groupby("Country")["InvoiceNo"]
    .nunique()
    .reset_index(name="Transaksi")
)

X = transaksi[["Transaksi"]].values
cluster = AgglomerativeClustering(n_clusters=3, linkage="average")
transaksi["Cluster"] = cluster.fit_predict(X)

centroid = transaksi.groupby("Cluster")["Transaksi"].mean()
sorted_centroid = centroid.sort_values(ascending=True)

cluster_order = sorted_centroid.index.tolist()
label_mapping = {
    cluster_order[0]: "Rendah",
    cluster_order[1]: "Sedang",
    cluster_order[2]: "Tinggi",
}
transaksi["Kategori"] = transaksi["Cluster"].map(label_mapping)

df_plot = transaksi.sort_values("Transaksi").reset_index(drop=True)

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
        sub["Transaksi"],
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
plt.ylabel("Transaksi")
plt.title("Visualisasi Clustering Transaksi per Negara (Average Linkage, k=3)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hasil_clustering_transaksi.png")
plt.savefig(output_path, dpi=300)
print(f"Grafik visualisasi disimpan di: {output_path}")

plt.show()
