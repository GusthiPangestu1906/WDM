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

for kat in ["Rendah", "Sedang", "Tinggi"]:
    sub = transaksi[transaksi["Kategori"] == kat]
    negara_list = sub["Country"].tolist()
    print(f"\nKategori Transaksi {kat} ({len(negara_list)} negara):")
    for _, row in sub.iterrows():
        print(f"  - {row['Country']}: {row['Transaksi']} transaksi")
