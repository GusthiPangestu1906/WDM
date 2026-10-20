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

print("Posisi centroid setiap cluster:")
for c_id, val in centroid.items():
    print(f"Cluster {c_id}: centroid = {val:.2f}")
