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

print("Pengurutan posisi centroid secara ascending:")
for rank, (c_id, val) in enumerate(sorted_centroid.items()):
    print(f"Urutan {rank} -> Cluster ID {c_id}: centroid = {val:.2f}")
