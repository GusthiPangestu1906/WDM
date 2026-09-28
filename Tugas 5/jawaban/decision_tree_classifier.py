import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_text

# Inisialisasi direktori
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')

# 1. Load dataset train, test, dan target
dataset = pd.read_csv(os.path.join(data_dir, 'titanic.csv'))
test_dataset = pd.read_csv(os.path.join(data_dir, 'titanic_test.csv'))
raw_test_label = pd.read_csv(os.path.join(data_dir, 'titanic_testlabel.csv'))

target_col = 'Survived' if 'Survived' in raw_test_label.columns else raw_test_label.columns[1]
test_label = raw_test_label[target_col]
train_label = dataset['Survived']

# 2. Ambil fitur yang digunakan
features = ['Sex', 'Age', 'Pclass', 'Fare']
train_data = dataset[features].copy()
test_data = test_dataset[features].copy()

# 3. Konversi fitur Sex menjadi numerik (0/1) agar bisa diproses model
gender_map = {'male': 0, 'female': 1}
train_data['Sex'] = train_data['Sex'].map(gender_map)
test_data['Sex'] = test_data['Sex'].map(gender_map)

# 4. Imputasi missing value Age & Fare
# Isi missing value Age train dengan mean tiap kelas Survived
train_data['Age'] = dataset.groupby('Survived')['Age'].transform(
    lambda x: x.fillna(x.mean())
)

# Isi missing value Age test berdasarkan kelas atau mean data train
test_data['Age'] = test_data.groupby(test_label)['Age'].transform(
    lambda x: x.fillna(x.mean())
)
test_data['Age'] = test_data['Age'].fillna(train_data['Age'].mean())

# Isi missing value Fare pada data test dengan nilai mean
test_data['Fare'] = test_data['Fare'].fillna(train_data['Fare'].mean())

# 5. Latih model perbandingan (Tanpa Pruning vs Dengan Pruning)
# Model 1: Default (Tanpa pembatasan kedalaman)
dtc_unpruned = DecisionTreeClassifier(random_state=42)
dtc_unpruned.fit(train_data, train_label)
acc_unpruned = dtc_unpruned.score(test_data, test_label)
err_unpruned = round((1 - acc_unpruned) * 100, 2)

# Model 2: Disederhanakan (Pre-pruning max_depth=3)
dtc = DecisionTreeClassifier(max_depth=3, random_state=42)
dtc.fit(train_data, train_label)

# 6. Lakukan prediksi & evaluasi pada data test
class_result = dtc.predict(test_data)
acc = dtc.score(test_data, test_label)
err = round((1 - acc) * 100, 2)

print("=" * 65)
print("PERBANDINGAN MODEL DECISION TREE PADA DATA TEST")
print("=" * 65)
print(f"{'Metrik / Model':<30} | {'Default':>14} | {'max_depth=3':>14}")
print("-" * 65)
print(f"{'Jumlah Total Node':<30} | {f'{dtc_unpruned.tree_.node_count} node':>14} | {f'{dtc.tree_.node_count} node':>14}")
print(f"{'Kedalaman Pohon (Depth)':<30} | {f'{dtc_unpruned.tree_.max_depth} level':>14} | {f'{dtc.tree_.max_depth} level':>14}")
print(f"{'Akurasi Pengujian (Accuracy)':<30} | {f'{round(acc_unpruned * 100, 2)}%':>14} | {f'{round(acc * 100, 2)}%':>14}")
print(f"{'Error Ratio':<30} | {f'{err_unpruned}%':>14} | {f'{err}%':>14}")
print("=" * 65)
print(f"Hasil Prediksi 10 sampel pertama (Pruned): {class_result[:10]}\n")

# 7. Informasi Ringkasan Node Pohon Keputusan
print("=" * 65)
print("INFORMASI STRUKTUR NODE POHON")
print("=" * 65)
print(f"{'Total Jumlah Node':<30}: {dtc.tree_.node_count} node")
print(f"{'Kedalaman Pohon (Depth)':<30}: {dtc.tree_.max_depth} level")
print(f"{'Jumlah Leaf Node (Daun)':<30}: {dtc.tree_.n_leaves} daun")
print("\nTingkat Kepentingan Fitur (Feature Importances):")
for feat, imp in zip(features, dtc.feature_importances_):
    print(f"  - {feat:<8}: {imp:.4f} ({imp*100:>5.2f}%)")

# 8. Tampilkan Struktur Node
print("\n" + "=" * 65)
print("STRUKTUR NODE")
print("=" * 65)
tree_rules = export_text(dtc, feature_names=features)
print(tree_rules)

# 9. Simpan Visualisasi Gambar Pohon Keputusan (.png via Matplotlib)
png_path = os.path.join(base_dir, "decision_tree.png")
fig, ax = plt.subplots(figsize=(18, 9), dpi=300)
tree.plot_tree(
    dtc,
    feature_names=features,
    class_names=['Not Survived (0)', 'Survived (1)'],
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax
)
plt.title("Visualisasi Struktur Node Decision Tree (max_depth=3) - Titanic Dataset", fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(png_path, bbox_inches='tight')
plt.close(fig)
print("=" * 65)
print(f"[OK] Gambar visualisasi node yang jelas berhasil disimpan ke:\n     {png_path}")

# 10. Simpan format .dot (Graphviz)
dot_path = os.path.join(base_dir, "decision_tree.dot")
dot_data = tree.export_graphviz(
    dtc,
    out_file=None,
    feature_names=features,
    class_names=['Not Survived', 'Survived'],
    filled=True,
    rounded=True
)
with open(dot_path, "w") as f:
    f.write(dot_data)
print(f"[OK] File .dot tersimpan di:\n     {dot_path}")
print("=" * 65)