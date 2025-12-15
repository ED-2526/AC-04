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
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from xgboost import XGBClassifier
from yellowbrick.cluster import KElbowVisualizer

# ==============================================================================
# 0. CONFIGURACIÓ INICIAL
# ==============================================================================
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# Crear carpeta de resultats
output_folders = {'silhouette': 'resultats_xgboost_silhouette'}

for folder in output_folders.values():
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    print(f"Carpeta '{folder}' preparada.")

# ==============================================================================
# 1. CÀRREGA I PREPROCESSAMENT DE DADES
# ==============================================================================
print("\n--- 1. Càrrega i Preprocessament de Dades ---")
filename = 'marketing_campaign.csv'

try:
    data = pd.read_csv(filename, sep="\t")
except FileNotFoundError:
    print(f"Error: No s'ha trobat '{filename}'.")
    exit()

# Feature Engineering Bàsic
data['Age'] = 2025 - data['Year_Birth']
data['Total_Spending'] = (
    data['MntWines'] + data['MntFruits'] +
    data['MntMeatProducts'] + data['MntFishProducts'] +
    data['MntSweetProducts'] + data['MntGoldProds']
)

# Eliminació Outliers Spending
Q1_spend = data['Total_Spending'].quantile(0.25)
Q3_spend = data['Total_Spending'].quantile(0.75)
IQR_spend = Q3_spend - Q1_spend
data = data[data['Total_Spending'] <= (Q3_spend + 2.0 * IQR_spend)]

# Variables Familiars
partner_status = ['Married', 'Together']
data['Has_Partner'] = data['Marital_Status'].apply(lambda x: 1 if x in partner_status else 0)
data['Family_Size'] = 1 + data['Has_Partner'] + data['Kidhome'] + data['Teenhome']

# Antiguitat Client
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], dayfirst=True)
data['Tenure_Days'] = (data['Dt_Customer'].max() - data['Dt_Customer']).dt.days

# Neteja Bàsica
data = data.dropna(subset=['Income'])
data = data[(data['Age'] < 100) & (data['Income'] < 600000)]
invalid_status = ['YOLO', 'Absurd', 'Alone']
data = data[~data['Marital_Status'].isin(invalid_status)]

# Outliers Income
Q1_inc = data['Income'].quantile(0.25)
Q3_inc = data['Income'].quantile(0.75)
data = data[data['Income'] <= (Q3_inc + 1.5 * (Q3_inc - Q1_inc))]

# One-Hot Encoding (per anàlisi posterior, no per clustering)
education_dummies = pd.get_dummies(data['Education'], prefix='Edu', drop_first=True)
marital_dummies = pd.get_dummies(data['Marital_Status'], prefix='Marital', drop_first=True)
data = pd.concat([data, education_dummies, marital_dummies], axis=1)

# Preferències de producte (spending ratios)
epsilon = 1e-6
data['Wine_Ratio'] = data['MntWines'] / (data['Total_Spending'] + epsilon)
data['Meat_Ratio'] = data['MntMeatProducts'] / (data['Total_Spending'] + epsilon)
data['Sweet_Ratio'] = data['MntSweetProducts'] / (data['Total_Spending'] + epsilon)
data['Fish_Ratio'] = data['MntFishProducts'] / (data['Total_Spending'] + epsilon)
data['Fruit_Ratio'] = data['MntFruits'] / (data['Total_Spending'] + epsilon)
data['Gold_Ratio'] = data['MntGoldProds'] / (data['Total_Spending'] + epsilon)

print(f"Dades netes: {len(data)} registres.")

# ==============================================================================
# 2. SELECCIÓ DE VARIABLES PER CLUSTERING
# ==============================================================================
selected_columns = [
    # Variables de comportament de compra
    'Income', 'Total_Spending', 
    'MntWines', 'MntMeatProducts', 'MntFishProducts',
    'MntFruits', 'MntSweetProducts', 'MntGoldProds',
    
    # Preferències de producte (ratios)
    'Wine_Ratio', 'Meat_Ratio', 'Sweet_Ratio', 
    'Fish_Ratio', 'Fruit_Ratio', 'Gold_Ratio',
    
    # Engagement i context
    'Tenure_Days', 'Family_Size', 'Age'
]

X = data[selected_columns].values
cols = selected_columns

print(f"\nVariables seleccionades per clustering ({len(cols)}):")
for i, col in enumerate(cols, 1):
    print(f"  {i}. {col}")

# ==============================================================================
# 3. DETERMINACIÓ K ÒPTIMA (MÈTODE DEL COLZE)
# ==============================================================================
print("\n--- 2. Determinant K Òptima amb Mètode del Colze ---")

scaler_elbow = StandardScaler()
X_scaled_elbow = scaler_elbow.fit_transform(X)

folder = output_folders['silhouette']
print(f"  Generant Elbow Plot (KMeans)...")

plt.figure(figsize=(10, 6))
visualizer = KElbowVisualizer(
    KMeans(random_state=42, n_init=10), 
    k=(2, 10), 
    metric='distortion',
    timings=False
)
visualizer.fit(X_scaled_elbow)
optimal_k = visualizer.elbow_value_
print(f"  K òptima detectada: {optimal_k}")
visualizer.show(outpath=os.path.join(folder, f'00_elbow_kmeans.png'))
plt.close()

# ==============================================================================
# 4. ANÀLISI D'IMPORTÀNCIA AMB XGBOOST
# ==============================================================================
print("\n--- 3. Entrenament XGBoost per Feature Selection ---")

def get_xgboost_importance(X_input, column_names, scaler_name, optimal_k):
    """
    Calcula la importància de features mitjançant XGBoost amb pseudo-labeling:
    1. Escala les dades
    2. Crea clusters temporals (KMeans) com a target
    3. Entrena XGBoost per predir aquests clusters
    4. Extreu les importàncies de features
    """
    scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
    X_scaled = scaler.fit_transform(X_input)
    
    # Pseudo-labeling: Clusters com a target artificial
    kmeans_base = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    y_pseudo = kmeans_base.fit_predict(X_scaled)
    
    # Entrenar XGBoost
    model = XGBClassifier(
        n_estimators=100, 
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    model.fit(X_scaled, y_pseudo)
    
    # Extreure importàncies
    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        'Feature': column_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return df_imp, model

# Calcular importàncies per ambdós scalers
print("  Calculant importàncies (Standard Scaler)...")
imp_standard, model_std = get_xgboost_importance(X, cols, 'standard', optimal_k)

print("  Calculant importàncies (MinMax Scaler)...")
imp_minmax, model_mm = get_xgboost_importance(X, cols, 'minmax', optimal_k)

# Guardar informes
txt_path = os.path.join(folder, 'resultats_xgboost_importance.txt')
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write(f"INFORME: XGBOOST FEATURE IMPORTANCE (Pseudo-Labeling K={optimal_k})\n")
    f.write("=========================================================\n\n")
    f.write("RÀNQUING (Standard Scaler):\n")
    f.write(imp_standard.to_string(index=False))
    f.write("\n\n")
    f.write("RÀNQUING (MinMax Scaler):\n")
    f.write(imp_minmax.to_string(index=False))

# Gràfic d'importància
plt.figure(figsize=(10, 6))
sns.barplot(data=imp_standard, x='Importance', y='Feature', palette='viridis')
plt.title('XGBoost Feature Importance (Standard Scaler)')
plt.tight_layout()
plt.savefig(os.path.join(folder, 'xgboost_feature_importance.png'))
plt.close()

print(f"  Informes guardats a {folder}")

# ==============================================================================
# 5. AVALUACIÓ CLUSTERING AMB LLINDARS XGBOOST
# ==============================================================================
print("\n--- 4. Avaluant Llindars d'Importància XGBoost ---")

def evaluate_xgboost_thresholds(X_original, column_names, imp_df_std, imp_df_minmax, 
                                 k_clusters, clustering_method='kmeans'):
    """
    Avalua diferents llindars d'importància XGBoost per selecció de variables.
    Retorna un DataFrame amb scores per cada combinació scaler-threshold.
    """
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
    scalers = ['minmax', 'standard']
    results = []
    
    for scaler_name in scalers:
        # Seleccionar importàncies segons scaler
        current_imp_df = imp_minmax if scaler_name == 'minmax' else imp_standard
        
        # Escalar dades
        scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
        X_scaled_global = scaler.fit_transform(X_original)
        
        for th in thresholds:
            # Seleccionar variables amb importància >= threshold
            selected_vars = current_imp_df[current_imp_df['Importance'] >= th]['Feature'].tolist()
            
            if len(selected_vars) < 2:
                continue
            
            selected_indices = [column_names.index(v) for v in selected_vars]
            X_subset = X_scaled_global[:, selected_indices]
            
            try:
                # Executar clustering
                if clustering_method == 'kmeans':
                    model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                elif clustering_method == 'gmm':
                    model = GaussianMixture(n_components=k_clusters, random_state=42)
                elif clustering_method == 'hierarchical':
                    model = AgglomerativeClustering(n_clusters=k_clusters)
                elif clustering_method == 'spectral':
                    model = SpectralClustering(n_clusters=k_clusters, random_state=42, 
                                              affinity='nearest_neighbors')
                
                labels = model.fit_predict(X_subset)
                
                # Calcular Silhouette
                if len(set(labels)) >= 2:
                    score = silhouette_score(X_subset, labels)
                    
                    # Penalització si hi ha variables categòriques
                    bad_keywords = ['Marital', 'Education', 'Edu']
                    has_bad_var = any(any(bk in var_name for bk in bad_keywords) 
                                     for var_name in selected_vars)
                    if has_bad_var:
                        score = score - 0.1
                else:
                    continue
                    
                results.append({
                    'scaler': scaler_name,
                    'threshold': th,
                    'num_vars': len(selected_vars),
                    'vars': ", ".join(selected_vars),
                    'score': score
                })
            except Exception:
                continue
            
    return pd.DataFrame(results)

# Processar només Silhouette
clustering_methods = ['kmeans', 'gmm', 'hierarchical', 'spectral']
metric_name = 'silhouette'
metric_config = {
    'folder': output_folders['silhouette'], 
    'ylabel': 'Silhouette Score', 
    'better': 'higher'
}

all_results = {metric_name: {}}

print(f"\n{'='*60}")
print(f"PROCESSANT MÈTRICA: {metric_name.upper()} (VIA XGBOOST)")
print(f"{'='*60}")

for method in clustering_methods:
    print(f"  Calculant {method.upper()} amb {metric_name}...")
    
    df_eval = evaluate_xgboost_thresholds(
        X, cols, imp_standard, imp_minmax, 
        k_clusters=optimal_k,
        clustering_method=method
    )
    
    all_results[metric_name][method] = df_eval
    
    # Guardar CSV
    csv_path = os.path.join(folder, f'evaluacio_xgboost_variables_{metric_name}_{method}.csv')
    df_eval.to_csv(csv_path, index=False)

print(f"\nCàlculs finalitzats. CSVs guardats.")

# ==============================================================================
# 6. GRÀFICS COMPARATIUS
# ==============================================================================
print("\n--- 5. Generant Gràfics Comparatius ---")

ylabel = metric_config['ylabel']

for method in clustering_methods:
    df_eval = all_results[metric_name][method]
    
    if df_eval.empty:
        print(f"    Warning: No dades per {method}")
        continue
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Gràfic Score vs Threshold
    sns.lineplot(data=df_eval, x='threshold', y='score', hue='scaler', 
                 marker='o', linewidth=2.5, palette=['blue', 'red'])
    
    plt.title(f'{ylabel} vs XGB Importance Threshold - {method.upper()}', fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('XGB Importance Threshold (> X)', fontsize=12)
    
    # Anotacions nombre de variables
    for i in range(df_eval.shape[0]):
        row = df_eval.iloc[i]
        plt.text(row['threshold'], row['score'], f"v={int(row['num_vars'])}", 
                 ha='center', va='bottom', size='small', weight='bold', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'grafic_comparativa_{metric_name}_{method}.png'))
    plt.close()
    
    print(f"    ✓ {method}")

# ==============================================================================
# 7. VISUALITZACIÓ MILLORS RESULTATS (PCA + t-SNE)
# ==============================================================================
print("\n--- 6. Visualitzant Millors Resultats amb PCA i t-SNE ---")

def get_best_result(df_results, method_name, metric_name, better='higher'):
    """Retorna la millor configuració amb mínim 2 variables"""
    if df_results.empty:
        return None
    
    df_filtered = df_results[df_results['num_vars'] >= 2]
    if df_filtered.empty:
        return None
    
    if better == 'higher':
        return df_filtered.loc[df_filtered['score'].idxmax()]
    else:
        return df_filtered.loc[df_filtered['score'].idxmin()]

def apply_clustering(X_subset, method, k_clusters=4):
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

def visualize_best_clustering(X_original, column_names, best_config, method_name, 
                              metric_name, folder):
    """Genera visualitzacions PCA i t-SNE per la millor configuració"""
    if best_config is None:
        return
    
    print(f"    - Scaler: {best_config['scaler']}")
    print(f"    - Threshold: {best_config['threshold']}")
    print(f"    - Variables: {best_config['vars']}")
    print(f"    - Score: {best_config['score']:.4f}")
    
    # Preparar dades
    scaler = MinMaxScaler() if best_config['scaler'] == 'minmax' else StandardScaler()
    X_scaled_full = scaler.fit_transform(X_original)
    
    # Seleccionar variables via XGBoost Importance
    imp_df = imp_minmax if best_config['scaler'] == 'minmax' else imp_standard
    selected_vars = imp_df[imp_df['Importance'] >= best_config['threshold']]['Feature'].tolist()
    selected_indices = [column_names.index(v) for v in selected_vars]
    
    if len(selected_indices) == 0:
        print(f"    [WARNING] No features selected. Skipping visualization.")
        return

    X_subset = X_scaled_full[:, selected_indices]
    labels = apply_clustering(X_subset, method_name, k_clusters=4)
    
    # Visualització
    n_viz_pca = min(2, X_subset.shape[1])
    pca_viz = PCA(n_components=n_viz_pca)
    X_pca = pca_viz.fit_transform(X_subset)
    
    plt.figure(figsize=(12, 5))
    
    # PCA Plot
    plt.subplot(1, 2, 1)
    if X_pca.shape[1] >= 2:
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', 
                              alpha=0.6, edgecolors='k', linewidth=0.5)
        plt.xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]*100:.1f}%)')
    else:
        scatter = plt.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), c=labels, 
                            cmap='viridis', alpha=0.6, edgecolors='k', linewidth=0.5)
        plt.xlabel('PC1')
        plt.ylabel('Fixed Axis')
        
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'PCA - {method_name.upper()}\nScore: {best_config["score"]:.4f}')
    plt.grid(True, alpha=0.3)
    
    # t-SNE Plot
    plt.subplot(1, 2, 2)
    perp = min(30, len(X_subset) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    X_tsne = tsne.fit_transform(X_subset)
    
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', 
                          alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(f't-SNE - {method_name.upper()}\nvars={len(selected_vars)}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'viz_best_{method_name}_xgboost_tsne.png'), dpi=150)
    plt.close()

# Processar visualitzacions
better = metric_config['better']

for method in clustering_methods:
    df_results = all_results[metric_name][method]
    best_config = get_best_result(df_results, method, metric_name, better)
    
    if best_config is not None:
        print(f"  {method.upper()}:")
        visualize_best_clustering(X, cols, best_config, method, metric_name, folder)

# ==============================================================================
# 8. PERFIL COMPLET DELS CLÚSTERS AMB MÈTRIQUES ADICIONALS
# ==============================================================================
print("\n--- 7. Perfil complet dels clústers (millor cas per mètode) ---")

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
        sse += np.sum(cdist(cluster_points, [centers[i]])**2)
    
    # SST (Total Sum of Squares)
    sst = np.sum((X_subset - global_mean)**2)
    
    # BSS (Between-cluster Sum of Squares)
    bss = sst - sse
    
    # Normalitzar per SST
    sse_norm = sse / sst
    bss_norm = bss / sst
    
    # Correlació (mostra si punts propers estan al mateix cluster)
    idx = np.random.choice(len(X_subset), min(len(X_subset), 1000), replace=False)
    X_sample = X_subset[idx]
    labels_sample = labels[idx]
    
    incidence_matrix = (labels_sample[:, None] == labels_sample[None, :]).astype(int)
    dist_matrix = pairwise_distances(X_sample)
    
    corr, _ = pearsonr(incidence_matrix.flatten(), dist_matrix.flatten())
    
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
    df_results = all_results[metric_name][method]
    best_config = get_best_result(df_results, method, metric_name, better)

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

    case_name = (f"XGBOOST BEST | {method.upper()} | "
                f"scaler={best_config['scaler']} | "
                f"th={best_config['threshold']} | k={optimal_k}")
    
    print_and_save_cluster_profile_full(data, best_labels, case_name, folder, extra_metrics)

# ==============================================================================
# RESUM FINAL
# ==============================================================================
print("\n" + "="*60)
print("PROCÉS COMPLETAT!")
print("="*60)
print(f"\n📁 Arxius generats a '{output_folders['silhouette']}':")
print("  ✓ Informe Feature Importance XGBoost (TXT)")
print("  ✓ Gràfic Feature Importance (PNG)")
print("  ✓ Mètode del colze per K òptima (PNG)")
print("  ✓ 4 gràfics comparatius amb SILHOUETTE (PNG)")
print("  ✓ 4 visualitzacions PCA+t-SNE millors resultats (PNG)")
print("  ✓ 4 perfils complets dels clústers amb mètriques (TXT + CSV)")
print("  ✓ CSVs amb avaluacions per cada mètode")
print("\n" + "="*60)