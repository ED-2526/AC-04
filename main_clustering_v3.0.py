# --- IMPORTS ---
import os
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn & Stats
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr

from yellowbrick.cluster import KElbowVisualizer

# Configuració
warnings.filterwarnings('ignore')

# --- 0. CONFIGURACIÓ DE CARPETES ---
output_folder = 'resultats_clustering'
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Neteja carpeta prèvia
os.makedirs(output_folder)
print(f"Carpeta '{output_folder}' preparada.")

# --- 1. DATA LOADING ---
print("\n--- 1. Loading Dataset ---")
filename = 'marketing_campaign.csv'

try:
    # El fitxer està separat per tabuladors (\t)
    data = pd.read_csv(filename, sep="\t")
    print(f"Fitxer '{filename}' carregat correctament.")
except FileNotFoundError:
    print("Error: No s'ha trobat el fitxer 'marketing_campaign.csv'.")
    exit()


# --- 2. PREPROCESSING & CLEANING ---
print("\n--- 2. Preprocessing & Cleaning ---")

# 2.1 Feature Engineering: Edat i Despesa
data['Age'] = 2025 - data['Year_Birth']
print(f"Clients inicials: {len(data)}")

data['Total_Spending'] = (
    data['MntWines'] + data['MntFruits'] +
    data['MntMeatProducts'] + data['MntFishProducts'] +
    data['MntSweetProducts'] + data['MntGoldProds']
)
data['Log_Spending'] = np.log1p(data['Total_Spending'])
# 2.2 Eliminació d'Outliers (Total_Spending)

print("-> Eliminant outliers de Despesa...")
Q1_spend = data['Total_Spending'].quantile(0.25)
Q3_spend = data['Total_Spending'].quantile(0.75)
IQR_spend = Q3_spend - Q1_spend
upper_bound_spend = Q3_spend + 1.5 * IQR_spend

outliers_spend = data[data['Total_Spending'] > upper_bound_spend]
print(f"   Eliminats {len(outliers_spend)} clients VIP extrems (> {upper_bound_spend:.2f}€).")
data = data[data['Total_Spending'] <= upper_bound_spend]

# 2.3 Feature Engineering: Família
partner_status = ['Married', 'Together']
data['Has_Partner'] = data['Marital_Status'].apply(lambda x: 1 if x in partner_status else 0)
data['Family_Size'] = 1 + data['Has_Partner'] + data['Kidhome'] + data['Teenhome']

"""# 2.4 Eliminació d'Outliers (Family_Size)
print("-> Eliminant outliers de Mida Familiar...")
Q1_fam = data['Family_Size'].quantile(0.25)
Q3_fam = data['Family_Size'].quantile(0.75)
IQR_fam = Q3_fam - Q1_fam
lower_bound_fam = Q1_fam - 1.5 * IQR_fam
upper_bound_fam = Q3_fam + 1.5 * IQR_fam

data = data[(data['Family_Size'] >= lower_bound_fam) & (data['Family_Size'] <= upper_bound_fam)]
print(f"   Clients restants després de neteja: {len(data)}")
"""
# 2.5 Feature Engineering: Antiguitat (Seniority)
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], dayfirst=True)
max_date = data['Dt_Customer'].max()
data['Tenure_Days'] = (max_date - data['Dt_Customer']).dt.days

data['Seniority'] = pd.cut(
    data['Tenure_Days'],
    bins=[-np.inf, 365,np.inf],
    labels=['Recent', 'Senior']
)
data['Seniority_Code'] = data['Seniority'].map({'Recent': 1, 'Senior': 2})
print("\nDistribució Seniority:\n", data['Seniority_Code'].value_counts())

# 2.6 Mapejos: Educació i Estat Civil
# Educació
education_map = {'Basic': 1, '2n Cycle': 2, 'Graduation': 3, 'Master': 4, 'PhD': 5}
data['Education_Code'] = data['Education'].map(education_map).fillna(0)

# Estat Civil (Agrupant Divorced/Widow/Single com a 3)
marital_status_map = {'Married': 1,'Together': 2,'Single': 3,'Divorced': 3,'Widow': 3}
data['Marital_Status_Code'] = data['Marital_Status'].map(marital_status_map).fillna(0)

# 2.7 Neteja Final de Valors Nuls i Absurds
data = data.dropna(subset=['Income', 'Education_Code'])
data = data[data['Age'] < 100]
data = data[data['Income'] < 600000]
invalid_status = ['YOLO', 'Absurd', 'Alone']
data = data[~data['Marital_Status'].isin(invalid_status)]

# 2.8 Eliminació d'Outliers (Income) - NORMALITZACIÓ ESTADÍSTICA
# Això elimina els punts que quedaven "flotant" al gràfic (els rics atípics)
Q1_inc = data['Income'].quantile(0.25)
Q3_inc = data['Income'].quantile(0.75)
IQR_inc = Q3_inc - Q1_inc

# Definim el límit superior
upper_bound_inc = Q3_inc + 1.5 * IQR_inc
# Comptem quants eliminem per informar
outliers_inc = data[data['Income'] > upper_bound_inc]
print(f"   Eliminats {len(outliers_inc)} clients amb ingressos anòmals (> {upper_bound_inc:.2f}€).")

# Apliquem el filtre
data = data[data['Income'] <= upper_bound_inc]




# --- 3. GUARDAR DADES PROCESSADES ---
output_csv = 'dades_processades_completes.csv'
data.to_csv(output_csv, index=False, sep=',')
print(f"\nCSV amb dades netes guardat: {output_csv}")

# --- 4. PREPARACIÓ PEL CLUSTERING ---
numerical_cols = [
    'Income', 'Total_Spending', 'Seniority_Code',
    'Education_Code', 'Marital_Status_Code'
]

X = data[numerical_cols].values
cols = list(data[numerical_cols].columns)
print(f"Matriu X preparada. Shape: {X.shape}")

# Escalat
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# --- 5. DETERMINACIÓ DE K ÒPTIMA (Mètode del Colze) ---
print("\n--- 5. Cerca de K Òptima ---")
model = KMeans(random_state=42, n_init=10)
visualizer = KElbowVisualizer(model, k=(2, 10), metric='distortion', timings=False)

visualizer.fit(X_scaled)
visualizer.show(outpath=f"{output_folder}/00_metode_colze.png")
optimal_k = visualizer.elbow_value_
print(f"K òptima detectada: {optimal_k}")

# Funció auxiliar per guardar gràfics
def save_2d_plot(X, labels, title, filename, method='pca'):
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    
    components = reducer.fit_transform(X)
    x, y = components[:, 0], components[:, 1]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=x, y=y, hue=labels, palette='viridis', s=80, alpha=0.8)
    plt.title(title)
    plt.legend(title='Clúster')
    plt.savefig(f'{output_folder}/{filename}')
    plt.close()
    print(f"-> Plot guardat: {filename}")
    

# --- 6. EXECUCIÓ DE MODELS ---
print(f"\n--- 6. Executant Models (K={optimal_k}) ---")

# 6.1 K-Means
print("Executant K-Means...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=1)
kmeans_labels = kmeans.fit_predict(X_scaled)
data['KMeans_Cluster'] = kmeans_labels
save_2d_plot(X_scaled, kmeans_labels, f'K-Means (K={optimal_k})', '01_kmeans_2d.png', method='tsne')

# 6.2 Jeràrquic
print("Executant Jeràrquic...")
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)
data['Hierarchical_Cluster'] = hierarchical_labels
save_2d_plot(X_scaled, hierarchical_labels, f'Jeràrquic (K={optimal_k})', '02_hierarchical_2d.png', method='tsne')

# 6.3 GMM
print("Executant GMM...")
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
data['GMM_Cluster'] = gmm_labels
save_2d_plot(X_scaled, gmm_labels, f'GMM (K={optimal_k})', '03_gmm_2d.png', method='tsne')


# --- 7. MÈTRIQUES ---
def print_metrics(X, labels, name):
    centers = np.array([X[labels == i].mean(axis=0) for i in range(optimal_k)])
    
    # SSE (Cohesió)
    sse = 0
    for i in range(optimal_k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            sse += np.sum(cdist(cluster_points, [centers[i]], metric='euclidean')**2)
            
    # BSS (Separació)
    bss = np.sum(pdist(centers, metric='euclidean'))
    
    # Correlació (mostra reduïda per velocitat)
    idx = np.random.choice(len(X), min(len(X), 1000), replace=False)
    X_s, L_s = X[idx], labels[idx]
    
    inc_mat = (L_s[:, None] == L_s[None, :]).astype(int)
    prox_mat = pairwise_distances(X_s)
    corr, _ = pearsonr(inc_mat.flatten(), prox_mat.flatten())
    
    print(f"{name}: SSE={sse:.2f}, BSS={bss:.2f}, Corr={corr:.4f}")

print("\n--- Resum de Mètriques ---")
print_metrics(X_scaled, kmeans_labels, "K-Means")
print_metrics(X_scaled, hierarchical_labels, "Jeràrquic")
print_metrics(X_scaled, gmm_labels, "GMM")


# --- 8. ANÀLISI PCA (Pesos) ---
pca = PCA(n_components=2)
pca.fit(X_scaled)
df_components = pd.DataFrame(
    pca.components_, 
    columns=numerical_cols, 
    index=['Component 1', 'Component 2']
)
print("\n--- Pesos del PCA ---")
print(df_components)

# --- 9. PERFILAT DETALLAT DELS CLÚSTERS ---
print("\n--- 9. Identificació i Profiling Complet ---")

# 1. Definim TOTES les columnes que volem analitzar
# (Incloses les que NO han entrat al model, com els productes o la recència)
profiling_cols = [
    # Demogràfiques
    'Income', 'Age', 'Family_Size', 'Seniority_Code', 
    'Education_Code', 'Marital_Status_Code',
    
    # Comportament i Valor
    'Total_Spending', 'Recency', 
    
    # Productes (El que volies veure)
    'MntWines', 'MntFruits', 'MntMeatProducts', 
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    
    # Canals
    'NumWebPurchases', 'NumCatalogPurchases', 
    'NumStorePurchases', 'NumWebVisitsMonth'
]

# 2. FILTRE DE SEGURETAT (La correcció de l'error)
# Seleccionem només les columnes que realment existeixen al DataFrame
existing_cols = [col for col in profiling_cols if col in data.columns]

# --- PAS CLAU: Convertim tot a números (float) ---
# Això evita l'error "category dtype does not support aggregation"
for col in existing_cols:
    try:
        data[col] = data[col].astype(float)
    except ValueError:
        print(f"Alerta: La columna {col} no s'ha pogut convertir a número i s'ignorarà.")
        existing_cols.remove(col)

# 3. Funció per imprimir el perfil transposat (més fàcil de llegir)
def print_cluster_profile(cluster_col_name, method_name):
    if cluster_col_name not in data.columns:
        return # Si el clúster no existeix, no fem res

    print(f"\n{'='*70}")
    print(f"📊  PERFIL COMPLET: {method_name.upper()}")
    print(f"{'='*70}")
    
    # Agrupem per clúster i calculem la mitjana
    # Com que ja hem forçat 'astype(float)' a dalt, això ja no fallarà
    summary = data.groupby(cluster_col_name)[existing_cols].mean().round(2)
    
    # Afegim el recompte de clients (quants n'hi ha a cada grup)
    summary['Count (Clients)'] = data[cluster_col_name].value_counts()
    
    # Transposem (.T) perquè les variables quedin a l'esquerra i els clústers a dalt
    print(summary.T)

print_cluster_profile('KMeans_Cluster', 'K-Means')

print_cluster_profile('Hierarchical_Cluster', 'Jeràrquic')

print_cluster_profile('GMM_Cluster', 'GMM')