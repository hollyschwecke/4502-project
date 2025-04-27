import os
import umap
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

base_path = '/Users/hollyschwecke/4502-project/csv_gsod'

# list to collect data
all_data = []

for year in range(2011, 2022):
    year_path = os.path.join(base_path, str(year))
    if not os.path.exists(year_path):
        continue
    
    for file in os.listdir(year_path):
        if file.endswith('.csv'):
            file_path = os.path.join(year_path, file)
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# combine data into one dataframe
if all_data:
    data = pd.concat(all_data, ignore_index=True)
else: 
    raise ValueError("No data was loaded. Check your file paths.")

# select numeric columns for umap
numeric_cols = data.select_dtypes(include=[np.number]).columns

# drop rows with missing numeric values
X = data[numeric_cols].dropna() 

# apply umap
reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='euclidean')
embedding = reducer.fit_transform(X)

# plt umap results
plt.figure(figsize=(10,0))
plt.scatter(embedding[:, 0], embedding[:, 1], s=2, alpha=0.6)
plt.title("UMMAP Projection of Weather Data")
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()