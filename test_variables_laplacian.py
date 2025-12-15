import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import warnings
import matplotlib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph
from yellowbrick.cluster import KElbowVisualizer
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

# ==============================================================================
# CONFIGURACIÓ INICIAL
# ==============================================================================
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# Suprimir warnings de fonts i altres warnings de matplotlib
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Crear carpeta de resultats
OUTPUT_FOLDER = 'resultats_laplacian_silhouette'
if os.path.exists(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)
os.makedirs(OUTPUT_FOLDER)
print(f"Carpeta '{OUTPUT_FOLDER}' preparada.\n")

# ==============================================================================
# 1. CÀRREGA I PREPROCESSAMENT DE DADES
# ==============================================================================
print("--- 1. Càrrega i Preprocessament de Dades ---")
filename = 'marketing_campaign.csv'

try:
    data = pd.read_csv(filename, sep="\t")
except FileNotFoundError:
    print(f"Error: No s'ha trobat '{filename}'.")
    exit()

# Feature Engineering
data['Age'] = 2025 - data['Year_Birth']
data['Total_Spending'] = (
    data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + 
    data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']
)

# Neteja d'outliers en Spending
Q1_spend = data['Total_Spending'].quantile(0.25)
Q3_spend = data['Total_Spending'].quantile(0.75)
IQR_spend = Q3_spend - Q1_spend
data = data[data['Total_Spending'] <= (Q3_spend + 2.0 * IQR_spend)]

# Variables familiars
partner_status = ['Married', 'Together']
data['Has_Partner'] = data['Marital_Status'].apply(lambda x: 1 if x in partner_status else 0)
data['Family_Size'] = 1 + data['Has_Partner'] + data['Kidhome'] + data['Teenhome']

# Antiguitat
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], dayfirst=True)
data['Tenure_Days'] = (data['Dt_Customer'].max() - data['Dt_Customer']).dt.days

# Neteja bàsica
data = data.dropna(subset=['Income'])
data = data[(data['Age'] < 100) & (data['Income'] < 600000)]
invalid_status = ['YOLO', 'Absurd', 'Alone']
data = data[~data['Marital_Status'].isin(invalid_status)]

# Outliers en Income
Q1_inc = data['Income'].quantile(0.25)
Q3_inc = data['Income'].quantile(0.75)
data = data[data['Income'] <= (Q3_inc + 1.5 * (Q3_inc - Q1_inc))]

# Variables de preferències (proporcions)
epsilon = 1e-6
data['Wine_Ratio'] = data['MntWines'] / (data['Total_Spending'] + epsilon)
data['Meat_Ratio'] = data['MntMeatProducts'] / (data['Total_Spending'] + epsilon)
data['Sweet_Ratio'] = data['MntSweetProducts'] / (data['Total_Spending'] + epsilon)
data['Fish_Ratio'] = data['MntFishProducts'] / (data['Total_Spending'] + epsilon)
data['Fruit_Ratio'] = data['MntFruits'] / (data['Total_Spending'] + epsilon)
data['Gold_Ratio'] = data['MntGoldProds'] / (data['Total_Spending'] + epsilon)

print(f"Dades netes: {len(data)} registres.\n")

# ==============================================================================
# 2. SELECCIÓ DE VARIABLES PER AL MODEL
# ==============================================================================
selected_columns = [
    # Comportament de compra
    'Income', 'Total_Spending', 
    'MntWines', 'MntMeatProducts', 'MntFishProducts',
    'MntFruits', 'MntSweetProducts', 'MntGoldProds',
    
    # Preferències (proporcions)
    'Wine_Ratio', 'Meat_Ratio', 'Sweet_Ratio', 
    'Fish_Ratio', 'Fruit_Ratio', 'Gold_Ratio',
    
    # Engagement i context
    'Tenure_Days', 'Family_Size', 'Age'
]

X = data[selected_columns].values
cols = selected_columns

print(f"Variables seleccionades ({len(cols)}):")
for i, col in enumerate(cols, 1):
    print(f"  {i}. {col}")

# ==============================================================================
# 3. DETERMINACIÓ DE K ÒPTIMA AMB ELBOW METHOD
# ==============================================================================
print("\n--- 2. Determinant K Òptima amb Elbow Method ---")

scaler_elbow = StandardScaler()
X_scaled_elbow = scaler_elbow.fit_transform(X)

plt.figure(figsize=(10, 6))
visualizer = KElbowVisualizer(
    KMeans(random_state=42, n_init=10), 
    k=(2, 10), 
    metric='distortion',
    timings=False
)
visualizer.fit(X_scaled_elbow)
optimal_k = visualizer.elbow_value_

print(f"K òptima segons Elbow Method: {optimal_k}")
visualizer.show(outpath=os.path.join(OUTPUT_FOLDER, '00_elbow_kmeans.png'))
plt.close()

# ==============================================================================
# 4. CÀLCUL DE LAPLACIAN SCORE (PSEUDO-IMPORTÀNCIA)
# ==============================================================================
print("\n--- 3. Càlcul de Laplacian Score Feature Importance ---")

def calculate_laplacian_importance(X_input, column_names, scaler_name, k_neighbors=10):
    """
    Calcula la importància de cada variable basant-se en el Laplacian Score.
    """
    # Escalat de dades
    scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
    X_scaled = scaler.fit_transform(X_input)
    
    # Construcció del graf de veïns
    try:
        affinity_matrix = kneighbors_graph(
            X_scaled, 
            n_neighbors=k_neighbors, 
            mode='connectivity',
            include_self=False
        ).toarray()
    except Exception as e:
        print(f"  Error en construir el graf: {e}")
        return pd.DataFrame({'Feature': column_names, 'Importance': [0.0] * len(column_names)})

    # Càlcul del Laplacian Score per cada variable
    D = np.diag(np.sum(affinity_matrix, axis=1))  # Matriu de grau
    L = D - affinity_matrix  # Laplacià (no normalitzat)
    
    scores = []
    for i in range(X_scaled.shape[1]):
        feature_vector = X_scaled[:, i].reshape(-1, 1)
        
        # Numerador: variació entre veïns
        numerator = np.dot(feature_vector.T, np.dot(L, feature_vector))[0, 0]
        
        # Denominador: normalització
        denominator = np.dot(feature_vector.T, np.dot(D, feature_vector))[0, 0]
        
        # Laplacian Score (menor és millor)
        laplacian_score = numerator / (denominator + 1e-6)
        scores.append(laplacian_score)
    
    # Convertir a importància (invertir i normalitzar)
    max_score = max(scores)
    min_score = min(scores)
    
    if max_score > min_score:
        importances = [(max_score - s) / (max_score - min_score) for s in scores]
    else:
        importances = [1.0] * len(scores)
    
    # Crear DataFrame ordenat
    df_imp = pd.DataFrame({
        'Feature': column_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return df_imp

# Calcular importàncies amb ambdós escaladors
print("  Calculant importàncies (Standard Scaler)...")
imp_standard = calculate_laplacian_importance(X, cols, 'standard')

print("  Calculant importàncies (MinMax Scaler)...")
imp_minmax = calculate_laplacian_importance(X, cols, 'minmax')

# Guardar informe
txt_path = os.path.join(OUTPUT_FOLDER, 'laplacian_importance.txt')
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("LAPLACIAN SCORE FEATURE IMPORTANCE\n")
    f.write("=" * 60 + "\n\n")
    f.write("STANDARD SCALER:\n")
    f.write(imp_standard.to_string(index=False))
    f.write("\n\n")
    f.write("MINMAX SCALER:\n")
    f.write(imp_minmax.to_string(index=False))

# Gràfic d'importància
plt.figure(figsize=(10, 6))
sns.barplot(data=imp_standard, x='Importance', y='Feature', palette='viridis')
plt.title('Laplacian Score Feature Importance (Standard Scaler)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'laplacian_importance.png'))
plt.close()

print(f"  Informe guardat: {txt_path}\n")

# ==============================================================================
# 5. AVALUACIÓ DE CLUSTERING AMB DIFERENTS LLINDARS
# ==============================================================================
print("--- 4. Avaluant Llindars d'Importància ---")

def evaluate_thresholds(X_original, column_names, imp_df_std, imp_df_minmax, k_clusters, method='kmeans'):
    """
    Avalua diferents llindars d'importància i retorna els resultats.
    """
    thresholds = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
    scalers = ['minmax', 'standard']
    results = []
    
    for scaler_name in scalers:
        current_imp = imp_minmax if scaler_name == 'minmax' else imp_df_std
        scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
        X_scaled = scaler.fit_transform(X_original)
        
        for th in thresholds:
            # Seleccionar variables amb importància >= threshold
            selected_vars = current_imp[current_imp['Importance'] >= th]['Feature'].tolist()
            
            if len(selected_vars) < 2:
                continue
            
            selected_indices = [column_names.index(v) for v in selected_vars]
            X_subset = X_scaled[:, selected_indices]
            
            try:
                # Aplicar clustering
                if method == 'kmeans':
                    model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                elif method == 'gmm':
                    model = GaussianMixture(n_components=k_clusters, random_state=42)
                elif method == 'hierarchical':
                    model = AgglomerativeClustering(n_clusters=k_clusters)
                elif method == 'spectral':
                    model = SpectralClustering(n_clusters=k_clusters, random_state=42, 
                                               affinity='nearest_neighbors')
                
                labels = model.fit_predict(X_subset)
                
                # Calcular Silhouette Score
                if len(set(labels)) >= 2:
                    score = silhouette_score(X_subset, labels)
                else:
                    continue
                
                results.append({
                    'scaler': scaler_name,
                    'threshold': th,
                    'num_vars': len(selected_vars),
                    'vars': ", ".join(selected_vars),
                    'silhouette': score
                })
            except:
                continue
    
    return pd.DataFrame(results)

# Executar per tots els mètodes de clustering
clustering_methods = ['kmeans', 'gmm', 'hierarchical', 'spectral']
all_results = {}

for method in clustering_methods:
    print(f"  Avaluant {method.upper()}...")
    df_eval = evaluate_thresholds(X, cols, imp_standard, imp_minmax, optimal_k, method)
    all_results[method] = df_eval
    
    # Guardar CSV
    csv_path = os.path.join(OUTPUT_FOLDER, f'evaluacio_{method}.csv')
    df_eval.to_csv(csv_path, index=False)

print("  Avaluacions completades.\n")

# ==============================================================================
# 6. GRÀFICS COMPARATIUS
# ==============================================================================
print("--- 5. Generant Gràfics Comparatius ---")

for method in clustering_methods:
    df_eval = all_results[method]
    
    if df_eval.empty:
        continue
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_eval, x='threshold', y='silhouette', hue='scaler', 
                 marker='o', linewidth=2.5, palette=['blue', 'red'])
    
    plt.title(f'Silhouette Score vs Laplacian Threshold - {method.upper()}', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.xlabel('Laplacian Score Threshold', fontsize=12)
    
    # Anotar nombre de variables
    for i in range(df_eval.shape[0]):
        row = df_eval.iloc[i]
        plt.text(row['threshold'], row['silhouette'], f"v={int(row['num_vars'])}", 
                 ha='center', va='bottom', size='small', weight='bold', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'comparativa_{method}.png'))
    plt.close()
    print(f"  ✓ {method}")

print()

# ==============================================================================
# 7. SELECCIÓ I VISUALITZACIÓ DEL MILLOR RESULTAT
# ==============================================================================
print("--- 6. Visualitzant Millors Resultats ---\n")

def get_best_result(df_results, method_name):
    """Retorna la millor configuració amb mínim 3 variables"""
    if df_results.empty:
        print(f"  Warning: No hi ha resultats per {method_name}")
        return None
    
    df_filtered = df_results[df_results['num_vars'] >= 3]
    
    if df_filtered.empty:
        print(f"  Warning: No hi ha resultats amb mínim 3 variables per {method_name}")
        return None
    
    best_row = df_filtered.loc[df_filtered['silhouette'].idxmax()]
    return best_row

def apply_clustering(X_subset, method, k_clusters):
    """Aplica el mètode de clustering especificat"""
    if method == 'kmeans':
        model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    elif method == 'gmm':
        model = GaussianMixture(n_components=k_clusters, random_state=42)
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=k_clusters)
    elif method == 'spectral':
        model = SpectralClustering(n_clusters=k_clusters, random_state=42, 
                                   affinity='nearest_neighbors')
    
    return model.fit_predict(X_subset)

def visualize_best(X_original, column_names, best_config, method_name, imp_std, imp_mm):
    """Visualitza el millor resultat amb PCA i t-SNE"""
    if best_config is None:
        return None
    
    print(f"  {method_name.upper()}:")
    print(f"    - Scaler: {best_config['scaler']}")
    print(f"    - Threshold: {best_config['threshold']}")
    print(f"    - Num Variables: {best_config['num_vars']}")
    print(f"    - Silhouette: {best_config['silhouette']:.4f}")
    print(f"    - Variables: {best_config['vars']}\n")
    
    # Preparar dades
    scaler = MinMaxScaler() if best_config['scaler'] == 'minmax' else StandardScaler()
    X_scaled = scaler.fit_transform(X_original)
    
    imp_df = imp_mm if best_config['scaler'] == 'minmax' else imp_std
    selected_vars = imp_df[imp_df['Importance'] >= best_config['threshold']]['Feature'].tolist()
    selected_indices = [column_names.index(v) for v in selected_vars]
    X_subset = X_scaled[:, selected_indices]
    
    # Generar labels
    labels = apply_clustering(X_subset, method_name, optimal_k)
    
    # Visualització
    pca = PCA(n_components=min(2, X_subset.shape[1]))
    X_pca = pca.fit_transform(X_subset)
    
    perp = min(30, len(X_subset) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    X_tsne = tsne.fit_transform(X_subset)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PCA
    if X_pca.shape[1] >= 2:
        scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', 
                                  alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    else:
        scatter = axes[0].scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), c=labels, 
                                  cmap='viridis', alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('Fixed Axis')
    
    axes[0].set_title(f'PCA - {method_name.upper()}\nSilhouette: {best_config["silhouette"]:.4f}')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Cluster')
    
    # t-SNE
    scatter = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', 
                              alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].set_title(f't-SNE - {method_name.upper()}\nVariables: {len(selected_vars)}')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label='Cluster')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'viz_best_{method_name}.png'), dpi=150)
    plt.close()
    
    return labels

# Visualitzar millors resultats
best_configs = {}
for method in clustering_methods:
    best_config = get_best_result(all_results[method], method)
    best_configs[method] = best_config
    visualize_best(X, cols, best_config, method, imp_standard, imp_minmax)


# ==============================================================================
# 8. PERFIL COMPLET DELS CLÚSTERS AMB MÈTRIQUES ADICIONALS
# ==============================================================================
print("--- 7. Perfil complet dels clústers (millor cas per mètode) ---\n")

def calculate_extra_metrics(X_subset, labels):
    """
    Calcula mètriques adicionals de qualitat del clustering:
    - SSE Normalitzat: Cohesió intra-cluster
    - BSS Normalitzat: Separació inter-cluster
    - Correlació Pearson: Relació incidència-proximitat
    """
    unique_labels = np.unique(labels)
    centers = np.array([X_subset[labels == i].mean(axis=0) for i in unique_labels])
    global_mean = X_subset.mean(axis=0)
    
    # SSE (Within-cluster Sum of Squares)
    sse = 0
    for i, label in enumerate(unique_labels):
        cluster_points = X_subset[labels == label]
        if len(cluster_points) > 0:
            sse += np.sum(cdist(cluster_points, [centers[i]])**2)
    
    # SST (Total Sum of Squares)
    sst = np.sum((X_subset - global_mean)**2)
    
    # BSS (Between-cluster Sum of Squares)
    bss = sst - sse
    
    # Normalitzar per SST
    sse_norm = sse / sst if sst > 1e-6 else 0
    bss_norm = bss / sst if sst > 1e-6 else 0
    
    # Correlació (mostra si punts propers estan al mateix cluster)
    if len(X_subset) < 2:
        corr = 0.0
    else:
        idx = np.random.choice(len(X_subset), min(len(X_subset), 1000), replace=False)
        X_sample = X_subset[idx]
        labels_sample = labels[idx]
        
        incidence_matrix = (labels_sample[:, None] == labels_sample[None, :]).astype(int)
        dist_matrix = pairwise_distances(X_sample)
        
        # Aplanar i calcular correlació
        inc_flat = incidence_matrix.flatten()
        dist_flat = dist_matrix.flatten()

        # Filtrar per evitar NaN/Inf si hi ha variància zero
        if np.all(inc_flat == inc_flat[0]) or np.all(dist_flat == dist_flat[0]):
            corr = 0.0
        else:
            try:
                corr, _ = pearsonr(inc_flat, dist_flat)
            except ValueError:
                corr = 0.0
    
    return sse_norm, bss_norm, corr

def print_and_save_cluster_profile_full(data_df, labels, case_name, out_folder, extra_metrics):
    """
    Imprimeix i guarda perfil complet dels clústers amb mètriques de qualitat
    """
    tmp = data_df.copy()
    tmp['_CLUSTER_'] = labels

    # Variables numèriques per perfil
    num_cols = tmp.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != '_CLUSTER_']

    summary = tmp.groupby('_CLUSTER_')[num_cols].mean().round(2)
    counts = tmp['_CLUSTER_'].value_counts().sort_index()
    summary['Count (Clients)'] = counts.values

    sse_n, bss_n, corr = extra_metrics

    # Header amb mètriques
    metrics_header = (
        f"\n{'='*90}\n"
        f"📊 PERFIL COMPLET: {case_name}\n"
        f"{'-'*90}\n"
        f"   >> BSS (Separació) Norm: {bss_n:.2%}  (Més alt millor)\n"
        f"   >> SSE (Cohesió) Norm:   {sse_n:.2%}  (Més baix millor)\n"
        f"   >> Correlació (Pearson): {corr:.4f}   (Més a prop de -1 millor)\n"
        f"{'='*90}"
    )

    print(metrics_header)
    print(summary.T)

    # Guardar arxius
    safe_name = case_name.lower().replace(" ", "_").replace("/", "_").replace("|", "")
    csv_path = os.path.join(out_folder, f'perfil_clusters_{safe_name}.csv')
    txt_path = os.path.join(out_folder, f'perfil_clusters_{safe_name}.txt')

    summary.T.to_csv(csv_path, sep='\t')

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(metrics_header + "\n\n")
        f.write(summary.T.to_string())
        f.write("\n")

    print(f"   -> Guardat: {csv_path}")
    print(f"   -> Guardat: {txt_path}")

def build_labels_and_metrics_for_best_config(X_original, column_names, best_config, 
                                             method_name, k_clusters):
    """
    Reprodueix la millor configuració i calcula labels + mètriques
    """
    if best_config is None:
        return None, None
        
    scaler = MinMaxScaler() if best_config['scaler'] == 'minmax' else StandardScaler()
    X_scaled_full = scaler.fit_transform(X_original)

    imp_df = imp_minmax if best_config['scaler'] == 'minmax' else imp_standard
    selected_vars = imp_df[imp_df['Importance'] >= best_config['threshold']]['Feature'].tolist()
    selected_indices = [column_names.index(v) for v in selected_vars]
    
    if len(selected_indices) == 0:
        return None, None

    X_subset = X_scaled_full[:, selected_indices]
    labels = apply_clustering(X_subset, method_name, k_clusters=k_clusters)
    extra_metrics = calculate_extra_metrics(X_subset, labels)
    
    return labels, extra_metrics


# Generar perfils per cada mètode
for method in clustering_methods:
    best_config = best_configs[method]

    if best_config is None:
        print(f"  ⚠️ Sense millor configuració per {method.upper()}.")
        continue

    best_labels, extra_metrics = build_labels_and_metrics_for_best_config(
        X_original=X,
        column_names=cols,
        best_config=best_config,
        method_name=method,
        k_clusters=optimal_k
    )
    
    if best_labels is None:
        print(f"  ⚠️ Error recalculant labels per {method.upper()}.")
        continue

    case_name = (f"LAPLACIAN BEST | {method.upper()} | "
                f"scaler={best_config['scaler']} | "
                f"th={best_config['threshold']} | k={optimal_k}")
    
    print_and_save_cluster_profile_full(data, best_labels, case_name, OUTPUT_FOLDER, extra_metrics)


# ==============================================================================
# RESUM FINAL
# ==============================================================================
print("=" * 60)
print("PROCÉS COMPLETAT!")
print("=" * 60)
print(f"\nResultats guardats a: '{OUTPUT_FOLDER}/'")
print("\nArxius generats:")
print("  ✓ Elbow Method per K òptima")
print("  ✓ Laplacian Score Feature Importance (TXT + PNG)")
print("  ✓ Avaluacions per cada mètode (CSV)")
print("  ✓ Gràfics comparatius (PNG)")
print("  ✓ Visualitzacions PCA + t-SNE (PNG)")
print("  ✓ Perfils complets dels clústers amb mètriques (CSV + TXT)")
print("=" * 60)