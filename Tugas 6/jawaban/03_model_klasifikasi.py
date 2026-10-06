import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from data_loader import load_heart_data, print_df

df, X, y = load_heart_data()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

models = {
    "k-NN (k=1)": KNeighborsClassifier(n_neighbors=1),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "k-NN (k=9)": KNeighborsClassifier(n_neighbors=9),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Latih dan evaluasi performa model pada data uji (Hold-out 70:30)
results = []
for name, clf in models.items():
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te) * 100
    results.append({"Model": name, "Akurasi (%)": f"{acc:.2f}%", "Error Ratio (%)": f"{100-acc:.2f}%"})

print_df(pd.DataFrame(results), title="Hasil Uji Coba Model (Baseline Split 70:30)")
