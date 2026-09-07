import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'titanic.csv')
if not os.path.exists(csv_path):
    csv_path = 'data/titanic.csv'

dataset = pd.read_csv(csv_path)

sns.set_theme(style="whitegrid")

# Visualisasi Urutan Data vs Sex
plt.figure(figsize=(10, 5))
sns.scatterplot(
    x=dataset.index, 
    y=dataset['Sex'], 
    hue=dataset['Survived'].map({0: 'Tidak Selamat (0)', 1: 'Selamat (1)'}),
    palette={'Tidak Selamat (0)': '#e74c3c', 'Selamat (1)': '#2ecc71'},
    alpha=0.7,
    s=50
)

plt.title('Visualisasi: Urutan Data vs Sex Berdasarkan Status Survived')
plt.xlabel('Urutan Data (Index)')
plt.ylabel('Sex')
plt.legend(title='Status Survived', loc='center right')
plt.tight_layout()

# Simpan ke folder output
output_dir = os.path.join(base_dir, '..', 'output')
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'visualisasi_no9_sex_survived.png')
plt.savefig(save_path, dpi=300)
print(f"Grafik disimpan di: {save_path}")

plt.show()
