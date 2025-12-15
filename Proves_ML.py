# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr
import warnings
import io
import os
import shutil
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')

# --- 0. CONFIGURACIÓ DE CARPETES ---
output_folder = 'resultats_clustering_2'
if os.path.exists(output_folder):
    shutil.rmtree(output_folder) # Neteja si ja existeix
os.makedirs(output_folder)
print(f"Carpeta '{output_folder}' creada per guardar els resultats.")

# --- 1. DATA LOADING ---
print("Attempting to load dataset...")
filename = 'marketing_campaign.csv'

try:
    data = pd.read_csv(filename, sep="\t")
    print(f"File '{filename}' loaded successfully!")
except FileNotFoundError:
    print(f"File not found. Please upload '{filename}'...")
    try:
        uploaded = files.upload()
        data = pd.read_csv(io.BytesIO(uploaded[filename]), sep="\t")
    except:
        print("Error loading file.")

# --- 2. PREPROCESSING & CLEANING ---
print("\n--- Preprocessing & Cleaning ---")
# Feature Engineering
data['Age'] = 2025 - data['Year_Birth']
data['Total_Spending'] = (data['MntWines'] + data['MntFruits'] + 
                          data['MntMeatProducts'] + data['MntFishProducts'] + 
                          data['MntSweetProducts'] + data['MntGoldProds'])
data['Family_Size'] = data['Kidhome'] + data['Teenhome']


# Neteja
data = data.dropna(subset=['Income'])
data = data[data['Age'] < 100]
data = data[data['Income'] < 600000]
invalid_status = ['YOLO', 'Absurd', 'Alone']
data = data[~data['Marital_Status'].isin(invalid_status)]
cat_cols = ["Education", "Marital_Status"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

# Ajustar + transformar
encoded_array = encoder.fit_transform(data[cat_cols])

encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(cat_cols),
    index=data.index
)

# Substituïm les categòriques originals per les dummies dins de 'data'
data = data.drop(columns=cat_cols)
data = pd.concat([data, encoded_df], axis=1)

# --- DEFINICIÓ DE COLUMNES ---

# Columnes numèriques
numeric_cols = [
    "Age",
    "Total_Spending",
    "Income",
    "Family_Size"
]

# Totes les dummies creades
onehot_cols = list(encoder.get_feature_names_out(cat_cols))

# Dummies de cada atribut categòric (per agrupar-les)
education_cols = [c for c in onehot_cols if c.startswith("Education_")]
marital_cols   = [c for c in onehot_cols if c.startswith("Marital_Status_")]

# Llista completa de columnes que entra a PCA / t-SNE
feature_cols = numeric_cols + onehot_cols

# Diccionari de GRUPS de features (el que vols)
feature_groups = {
    "Age": ["Age"],
    "Total_Spending": ["Total_Spending"],
    "Income": ["Income"],
    "Family_Size": ["Family_Size"],
    "Education": education_cols,
    "Marital_Status": marital_cols
}

# Matriu X final
X = data[feature_cols].copy()


# --- 2. Escalat de les variables ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. PCA per importància de variables ---
n_components = min(len(feature_cols), 5)
pca = PCA(n_components=n_components)
pca.fit(X_scaled)

components = pca.components_
explained_var = pca.explained_variance_ratio_

# Importància per COLUMNA (números i dummies)
col_importance_array = np.zeros(len(feature_cols))
for i_comp in range(len(explained_var)):
    col_importance_array += np.abs(components[i_comp, :]) * explained_var[i_comp]

col_importance = pd.Series(col_importance_array, index=feature_cols)

# --- AGRUPAR PER ATRIBUT (Education, Marital_Status, etc.) ---
group_rows = []
for group_name, cols in feature_groups.items():
    group_rows.append({
        "feature_group": group_name,
        "importance": col_importance[cols].sum()
    })

pca_importance_grouped_df = pd.DataFrame(group_rows).sort_values(
    "importance", ascending=False
)

print("Importància de variables segons PCA (agrupades per atribut):")
print(pca_importance_grouped_df)






# --- 3. t-SNE base (2D) ---
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    random_state=42,
    init="pca"  # ajuda a estabilitzar una mica
)

tsne_base = tsne.fit_transform(X_scaled)

# --- 3b. Gràfic t-SNE base ---
plt.figure(figsize=(8, 6))
plt.scatter(tsne_base[:, 0], tsne_base[:, 1], s=20, alpha=0.7)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("Projecció t-SNE 2D de les dades")

tsne_plot_path = os.path.join(output_folder, "tsne_scatter.png")
plt.savefig(tsne_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Gràfic t-SNE guardat a: {tsne_plot_path}")

# --- 4. Funció per comparar embeddings ---
def embedding_difference(E1, E2):
    D1 = pairwise_distances(E1)
    D2 = pairwise_distances(E2)
    return np.mean(np.abs(D1 - D2))

# --- 5. Impacte per ATRIBUT (t-SNE perturbation importance, agrupat) ---
results = []

for group_name, cols in feature_groups.items():
    # Eliminem totes les columnes d'aquest atribut (p.ex. totes les Education_*)
    reduced_cols = [c for c in feature_cols if c not in cols]

    X_reduced = data[reduced_cols].copy()
    X_reduced_scaled = StandardScaler().fit_transform(X_reduced)

    tsne_reduced = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        random_state=42,
        init="pca"
    ).fit_transform(X_reduced_scaled)

    diff = embedding_difference(tsne_base, tsne_reduced)

    results.append({
        "feature_group": group_name,
        "importance": diff
    })

tsne_importance_grouped_df = pd.DataFrame(results).sort_values(
    "importance", ascending=False
)

print("Importància de variables segons t-SNE (agrupades per atribut):")
print(tsne_importance_grouped_df)

# Opcional: guardar la taula d'importància agrupada
tsne_importance_path = os.path.join(output_folder, "tsne_feature_importance_grouped.csv")
tsne_importance_grouped_df.to_csv(tsne_importance_path, index=False)
print(f"Taula d'importància t-SNE (grups) guardada a: {tsne_importance_path}")

