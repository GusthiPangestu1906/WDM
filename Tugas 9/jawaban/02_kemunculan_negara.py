from data_loader import load_dataset

dataset = load_dataset()

country = dataset["Country"].value_counts().reset_index()
country.columns = ["Country", "Kemunculan"]

print("Kemunculan tiap negara pada dataset:")
print(country.to_string(index=False))
