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

print("Hasil clustering Average Linkage (k=3):")
print(transaksi[["Country", "Transaksi", "Cluster"]].to_string(index=False))
