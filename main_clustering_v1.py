# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D # Necessari per al 3D
import warnings
import io
import os
import shutil

warnings.filterwarnings('ignore')

# --- 0. CONFIGURACIÓ DE CARPETES ---
# Crea una carpeta per guardar tots els resultats del clustering
output_folder = 'resultats_clustering'
if os.path.exists(output_folder):
    shutil.rmtree(output_folder) # Neteja la carpeta si ja existeix per evitar confusions
os.makedirs(output_folder)
print(f"Carpeta '{output_folder}' creada per guardar els resultats.")

# --- 1. DATA LOADING ---
# Carrega el fitxer CSV amb les dades de la campanya de màrqueting
print("Attempting to load dataset...")
filename = 'marketing_campaign.csv'

try:
    # El fitxer està separat per tabuladors (\t) en lloc de comes
    data = pd.read_csv(filename, sep="\t")
    print(f"File '{filename}' loaded successfully!")
except FileNotFoundError:
    print("Error: File not found. Please ensure 'marketing_campaign.csv' is in the same directory.")

# --- 2. PREPROCESSING & CLEANING ---
print("\n--- Preprocessing & Cleaning ---")

# Feature Engineering: Crea noves variables a partir de les existents
data['Age'] = 2025 - data['Year_Birth']  # Calcula l'edat dels clients
data['Total_Spending'] = (data['MntWines'] + data['MntFruits'] + 
                          data['MntMeatProducts'] + data['MntFishProducts'] + 
                          data['MntSweetProducts'] + data['MntGoldProds'])
# Suma totes les despeses en diferents categories de productes

# Neteja de dades: elimina valors problemàtics
data = data.dropna(subset=['Income'])  # Elimina files sense dades d'ingressos
data = data[data['Age'] < 100]  # Elimina edats poc realistes (més de 100 anys)
data = data[data['Income'] < 600000]  # Elimina ingressos extremadament alts (outliers)
invalid_status = ['YOLO', 'Absurd', 'Alone']
data = data[~data['Marital_Status'].isin(invalid_status)]  # Elimina estats civils invàlids

# Selecció de variables per al clustering i escalat de dades
numerical_cols = ['Age', 'Income', 'Total_Spending']  # Les tres variables que usarem
X = data[numerical_cols].values  # Converteix a array de NumPy
robust_scaler = RobustScaler()  # Escalador robust als outliers
X_scaled = robust_scaler.fit_transform(X)  # Estandarditza les dades

# --- 3. TROBAR K ÒPTIMA ---
# Utilitza el mètode del colze per determinar el nombre òptim de clústers
print("\n--- Cerca de K Òptima ---")
inertia = []  # Guardarà la inèrcia (suma de distàncies al quadrat) per cada K
k_range = range(2, 11)  # Prova de 2 a 10 clústers

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)  # Guarda la inèrcia d'aquest K

# Gràfic del Colze: mostra com disminueix la inèrcia amb més clústers
plt.figure(figsize=(10, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Mètode del Colze')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig(f'{output_folder}/00_metode_colze.png')
plt.close()

optimal_k = 4  # Selecció manual basada en l'anàlisi visual del gràfic
print(f"La K òptima seleccionada és: {optimal_k}")

# --- FUNCIÓ PER GRAFICAR EN 3D ---
def save_3d_plot(df, labels, title, filename):
    """
    Crea un gràfic 3D dels clústers i el guarda com a imatge.
    Paràmetres:
    - df: DataFrame amb les dades
    - labels: etiquetes dels clústers per cada punt
    - title: títol del gràfic
    - filename: nom del fitxer per guardar
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')  # Crea els eixos 3D

    # Defineix les tres dimensions del gràfic
    x = df['Age']  # Eix X: Edat
    y = df['Income']  # Eix Y: Ingressos
    z = df['Total_Spending']  # Eix Z: Despesa Total
    
    # Dibuixa els punts amb colors segons el clúster
    try:
        scatter = ax.scatter(x, y, z, c=labels, cmap='viridis', s=60, alpha=0.6, edgecolors='w')
        
        # Afegeix una llegenda per identificar els clústers
        legend1 = ax.legend(*scatter.legend_elements(), title="Clústers")
        ax.add_artist(legend1)
    except:
        pass  # Si hi ha algun error, continua sense trencar-se

    # Etiquetes dels eixos
    ax.set_xlabel('Edat (Anys)')
    ax.set_ylabel('Ingressos (€)')
    ax.set_zlabel('Despesa Total (€)')
    ax.set_title(title)

    # Guarda el gràfic
    plt.savefig(f'{output_folder}/{filename}')
    plt.close()
    print(f"-> Gràfic 3D guardat: {filename}")


# --- 4. EXECUCIÓ I GUARDAT DE MODELS ---
# Aplica tres algoritmes de clustering diferents i guarda els resultats
print(f"\n--- Executant i Guardant Models (K={optimal_k}) ---")

# 4.1 K-Means: mètode de partició que minimitza la distància als centroides
print("Processant K-Means...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)  # Assigna cada punt a un clúster
data['KMeans_Cluster'] = kmeans_labels  # Guarda les etiquetes al DataFrame
save_3d_plot(data, kmeans_labels, f'K-Means 3D (K={optimal_k})', '01_kmeans_3d.png')


# 4.2 Jeràrquic: crea una jerarquia de clústers (bottom-up)
print("Processant Jeràrquic...")
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)
data['Hierarchical_Cluster'] = hierarchical_labels
save_3d_plot(data, hierarchical_labels, f'Hierarchical 3D (K={optimal_k})', '02_hierarchical_3d.png')


# 4.3 GMM (Gaussian Mixture Model): model probabilístic amb distribucions gaussianes
print("Processant GMM...")
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
data['GMM_Cluster'] = gmm_labels
save_3d_plot(data, gmm_labels, f'GMM 3D (K={optimal_k})', '03_gmm_3d.png')

# --- 6. CÀLCUL DE MÈTRIQUES (Imprimir per pantalla) ---
def print_metrics(X, labels, name):
    """
    Calcula i imprimeix mètriques de qualitat del clustering:
    - SSE (Within-cluster sum of squares): mesura la cohesió interna
    - BSS (Between-cluster sum of squares): mesura la separació entre clústers
    - Correlació: relació entre proximitat i pertinença al mateix clúster
    """
    # Calcula els centres de cada clúster
    centers = np.array([X[labels == i].mean(axis=0) for i in range(optimal_k)])
    
    # SSE: suma de distàncies al quadrat dins de cada clúster (més baix = millor)
    sse = 0 
    for i in range(optimal_k):
        cluster_points = X[labels == i]  # Punts del clúster i
        if len(cluster_points) > 0:
            sse += np.sum(cdist(cluster_points, [centers[i]], metric='euclidean')**2)
    
    # BSS: suma de distàncies entre centres (més alt = millor separació)
    bss = np.sum(pdist(centers, metric='euclidean'))
    
    # Correlació: mesura si punts propers estan al mateix clúster
    # Usa una mostra per eficiència si hi ha moltes dades
    if len(X) > 1000:
        idx = np.random.choice(len(X), 1000, replace=False)
        X_s = X[idx]; L_s = labels[idx]
    else:
        X_s = X; L_s = labels
    
    # Matriu d'incidència (1 si mateix clúster, 0 si diferent)
    inc_mat = (L_s[:, None] == L_s[None, :]).astype(int)
    # Matriu de proximitat (distàncies entre punts)
    prox_mat = pairwise_distances(X_s)
    # Correlació de Pearson entre les dues matrius
    corr, _ = pearsonr(inc_mat.flatten(), prox_mat.flatten())
    
    print(f"{name}: SSE(Cohesió)={sse:.2f}, BSS(Separació)={bss:.2f}, Correlació={corr:.4f}")

# Imprimeix les mètriques per comparar els tres models
print("\n--- Resum de Mètriques ---")
print_metrics(X_scaled, kmeans_labels, "K-Means")
print_metrics(X_scaled, hierarchical_labels, "Jeràrquic")
print_metrics(X_scaled, gmm_labels, "GMM")

# --- 7. GUARDAR CSV ---
# Guarda el DataFrame amb les etiquetes dels clústers de tots els models
csv_path = f'{output_folder}/marketing_campaign_final.csv'
data.to_csv(csv_path, index=False, sep='\t')
print(f"\nCSV guardat a: {csv_path}")
print("Processament completat.")