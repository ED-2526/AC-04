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
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from yellowbrick.cluster import KElbowVisualizer
# ==============================================================================
# 0. CONFIGURACIÓ INICIAL
# ==============================================================================
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# Suprimir warnings de fonts i altres warnings de matplotlib
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

output_folder = 'resultats_clustering'
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

print(f"Carpeta '{output_folder}' preparada.")

txt_output_path = os.path.join(output_folder, 'resultats_pca_clustering_detallats.txt')

# ==============================================================================
# 1. CÀRREGA I NETEJA DE DADES
# ==============================================================================
print("\n--- 1. Càrrega i Preprocessament de Dades ---")
filename = 'marketing_campaign.csv'

try:
    data = pd.read_csv(filename, sep="\t")
except FileNotFoundError:
    print(f"Error: No s'ha trobat '{filename}'.")
    exit()

# --- Feature Engineering ---
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

# Antiguitat
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], dayfirst=True)
data['Tenure_Days'] = (data['Dt_Customer'].max() - data['Dt_Customer']).dt.days
data['Seniority_Code'] = pd.cut(data['Tenure_Days'], bins=[-np.inf, 365, np.inf], labels=[1, 2]).astype(int)

# Mapejos
education_map = {'Basic': 1, '2n Cycle': 2, 'Graduation': 3, 'Master': 4, 'PhD': 5}
data['Education_Code'] = data['Education'].map(education_map).fillna(0)
marital_map = {'Married': 1, 'Together': 2, 'Divorced': 3, 'Widow': 4, 'Single': 5}
data['Marital_Status_Code'] = data['Marital_Status'].map(marital_map).fillna(0)

# Neteja Final
data = data.dropna(subset=['Income', 'Education_Code'])
data = data[(data['Age'] < 100) & (data['Income'] < 600000)]
invalid_status = ['YOLO', 'Absurd', 'Alone']
data = data[~data['Marital_Status'].isin(invalid_status)]

# Outliers Income
Q1_inc = data['Income'].quantile(0.25)
Q3_inc = data['Income'].quantile(0.75)
data = data[data['Income'] <= (Q3_inc + 1.5 * (Q3_inc - Q1_inc))]

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
# 2. SELECCIÓ DE VARIABLES
# ==============================================================================
selected_columns = [
    # SPENDING BEHAVIOR
    'Income', 'Total_Spending', 
    'MntWines', 'MntMeatProducts', 'MntFishProducts',
    'MntFruits', 'MntSweetProducts', 'MntGoldProds',
    
    # PREFERÈNCIES
    'Wine_Ratio', 'Meat_Ratio', 'Sweet_Ratio', 'Fish_Ratio', 'Fruit_Ratio', 'Gold_Ratio',
    
    # ENGAGEMENT
    'Tenure_Days', 
    
    # CONTEXT FAMILIAR
    'Family_Size', 'Age'
]

X = data[selected_columns].values
cols = selected_columns

# ==============================================================================
# 3. DETERMINACIÓ DE COMPONENTS ÒPTIMS PCA
# ==============================================================================
def find_optimal_components(X_input, scaler_name, variance_threshold=0.90):
    """
    Determina el nombre mínim de components PCA necessaris per superar
    el llindar de variància especificat (per defecte 90%).
    
    Args:
        X_input: Matriu de dades
        scaler_name: 'minmax' o 'standard'
        variance_threshold: Llindar de variància acumulada (0.0-1.0)
    
    Returns:
        int: Nombre òptim de components
    """
    scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
    X_scaled = scaler.fit_transform(X_input)
    
    pca_full = PCA().fit(X_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    optimal_n = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    print(f"  [{scaler_name.upper()}] Components òptims: {optimal_n} "
          f"(variància: {cumulative_variance[optimal_n-1]*100:.2f}%)")
    
    return optimal_n

# ==============================================================================
# 4. DETERMINACIÓ DE K ÒPTIM AMB ELBOW METHOD
# ==============================================================================
def find_optimal_k(X_input, scaler_name, max_k=10):
    """
    Determina el nombre òptim de clusters utilitzant el mètode Elbow
    amb Yellowbrick per visualització.
    
    Args:
        X_input: Matriu de dades escalades
        scaler_name: Nom del scaler (per identificació)
        max_k: Nombre màxim de clusters a provar
    
    Returns:
        int: Nombre òptim de clusters
    """
    model = KMeans(random_state=42, n_init=10)
    visualizer = KElbowVisualizer(model, k=(2, max_k+1), timings=False)
    
    visualizer.fit(X_input)
    optimal_k = visualizer.elbow_value_
    
    # Guardar visualització
    visualizer.fig.savefig(os.path.join(output_folder, f'elbow_plot_{scaler_name}.png'))
    plt.close()
    
    if optimal_k is None:
        print(f"  [{scaler_name.upper()}] Warning: No s'ha detectat elbow clar, usant k=4 per defecte")
        optimal_k = 4
    else:
        print(f"  [{scaler_name.upper()}] K òptim detectat: {optimal_k}")
    
    return optimal_k

# ==============================================================================
# 5. ANÀLISI PCA DETALLADA
# ==============================================================================
def run_pca_analysis(X_input, scaler_name, n_components, column_names, file_handle):
    """
    Executa anàlisi PCA complet amb scree plot i loadings.
    
    Args:
        X_input: Matriu de dades
        scaler_name: 'minmax' o 'standard'
        n_components: Nombre de components a retenir
        column_names: Noms de les variables
        file_handle: Fitxer on escriure els resultats
    """
    title = f"ANÀLISI PCA: {scaler_name.upper()}"
    file_handle.write(f"\n{'#'*60}\n{title}\n{'#'*60}\n")
    
    scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
    X_scaled = scaler.fit_transform(X_input)
    
    # Desviació Estàndard
    file_handle.write("\n[1] Desviació Estàndard:\n")
    std_s = pd.Series(X_scaled.std(axis=0), index=column_names).sort_values(ascending=False)
    for k, v in std_s.items(): 
        file_handle.write(f"{k:20}: {v:.4f}\n")

    # Scree Plot
    pca_full = PCA().fit(X_scaled)
    var = pca_full.explained_variance_ratio_ * 100
    cumulative_var = np.cumsum(var)
    
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(var)+1), var, marker='o', color='black', label='Variància Individual')
    plt.bar(range(1, len(var)+1), var, alpha=0.7)
    plt.plot(range(1, len(var)+1), cumulative_var, marker='s', color='red', 
             linestyle='--', label='Variància Acumulada')
    plt.axhline(y=90, color='green', linestyle=':', linewidth=2, label='Llindar 90%')
    plt.title(f'Scree Plot ({scaler_name})')
    plt.xlabel('Nombre de Components')
    plt.ylabel('Variància Explicada (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_folder, f'scree_plot_{scaler_name}.png'))
    plt.close()

    # PCA Final amb n_components òptims
    n_comps = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=n_comps).fit(X_scaled)
    
    file_handle.write(f"\n[2] Variança Acumulada ({n_comps} comps): {pca.explained_variance_ratio_.cumsum()[-1]*100:.2f}%\n")
    
    # Loadings
    comps_df = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(n_comps)], index=column_names)
    file_handle.write("\n[3] PCA Loadings (Variables més importants per component):\n")
    for i in range(n_comps):
        pc = f"PC{i+1}"
        top = comps_df[pc].abs().sort_values(ascending=False)
        file_handle.write(f"\n>> {pc} ({pca.explained_variance_ratio_[i]*100:.2f}%):\n")
        for vname, val in top.items():
            real_val = comps_df.loc[vname, pc]
            file_handle.write(f"   - {vname:20} : {real_val:+.4f}\n")

# ==============================================================================
# 6. AVALUACIÓ CLUSTERING AMB DIFERENTS LLINDARS
# ==============================================================================
def evaluate_thresholds(X_original, column_names, clustering_method='kmeans', 
                       optimal_k_dict=None, optimal_pca_dict=None):
    """
    Avalua diferents llindars de selecció de variables amb diversos mètodes de clustering.
    Utilitza els valors òptims de K i components PCA calculats prèviament.
    
    Args:
        X_original: Matriu de dades original
        column_names: Noms de les variables
        clustering_method: Mètode de clustering a utilitzar
        optimal_k_dict: Diccionari amb k òptim per cada scaler
        optimal_pca_dict: Diccionari amb components PCA òptims per cada scaler
    
    Returns:
        DataFrame amb resultats de l'avaluació
    """
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    scalers = ['minmax', 'standard']
    
    results = []
    
    for scaler_name in scalers:
        # Utilitzar els valors òptims calculats prèviament
        k_clusters = optimal_k_dict.get(scaler_name, 4)
        n_pca_comps = optimal_pca_dict.get(scaler_name, 4)
        
        scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
        X_scaled_global = scaler.fit_transform(X_original)
        
        # Calcular loadings amb el nombre òptim de components
        pca = PCA(n_components=n_pca_comps)
        pca.fit(X_scaled_global)
        loadings_abs = np.abs(pca.components_)
        max_loading_per_feature = np.max(loadings_abs, axis=0)
        
        for th in thresholds:
            selected_indices = np.where(max_loading_per_feature >= th)[0]
            
            if len(selected_indices) < 2:
                continue
            
            current_vars = [column_names[i] for i in selected_indices]
            X_subset = X_scaled_global[:, selected_indices]
            
            try:
                # Aplicar el mètode de clustering seleccionat
                if clustering_method == 'kmeans':
                    model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                    labels = model.fit_predict(X_subset)
                    
                elif clustering_method == 'gmm':
                    model = GaussianMixture(n_components=k_clusters, random_state=42)
                    labels = model.fit_predict(X_subset)
                    
                elif clustering_method == 'hierarchical':
                    model = AgglomerativeClustering(n_clusters=k_clusters)
                    labels = model.fit_predict(X_subset)
                        
                elif clustering_method == 'spectral':
                    model = SpectralClustering(n_clusters=k_clusters, random_state=42, 
                                              affinity='nearest_neighbors')
                    labels = model.fit_predict(X_subset)
                
                # Calcular silhouette score
                if len(set(labels)) >= 2:
                    sil_score = silhouette_score(X_subset, labels)
                else:
                    continue
                    
                results.append({
                    'scaler': scaler_name,
                    'threshold': th,
                    'num_vars': len(current_vars),
                    'vars': ", ".join(current_vars),
                    'silhouette': sil_score,
                    'k_clusters': k_clusters
                })
                
            except Exception as e:
                print(f"  Warning [{clustering_method}]: Error amb threshold={th}, scaler={scaler_name}")
                continue
            
    return pd.DataFrame(results)

# ==============================================================================
# 7. VISUALITZACIÓ MILLORS RESULTATS
# ==============================================================================
def get_best_result(df_results, method_name, min_vars=3):
    """
    Retorna la millor configuració (màxim silhouette score) amb mínim nombre de variables.
    
    Args:
        df_results: DataFrame amb resultats
        method_name: Nom del mètode de clustering
        min_vars: Nombre mínim de variables requerides
    
    Returns:
        Series amb la millor configuració o None
    """
    if df_results.empty:
        print(f"  Warning: No hi ha resultats per {method_name}")
        return None
    
    df_filtered = df_results[df_results['num_vars'] >= min_vars]
    
    if df_filtered.empty:
        print(f"  Warning: No hi ha resultats amb mínim {min_vars} variables per {method_name}")
        return None
    
    best_row = df_filtered.loc[df_filtered['silhouette'].idxmax()]
    return best_row

def apply_clustering(X_subset, method, k_clusters=4):
    """
    Aplica el mètode de clustering especificat.
    
    Args:
        X_subset: Matriu de dades
        method: Mètode de clustering
        k_clusters: Nombre de clusters
    
    Returns:
        array: Etiquetes de cluster
    """
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
    return labels

def visualize_best_clustering(X_original, column_names, best_config, method_name, 
                              optimal_pca_dict):
    """
    Crea visualitzacions PCA i t-SNE per la millor configuració de clustering.
    
    Args:
        X_original: Matriu de dades original
        column_names: Noms de les variables
        best_config: Millor configuració trobada
        method_name: Nom del mètode de clustering
        optimal_pca_dict: Components PCA òptims per cada scaler
    """
    if best_config is None:
        return
    
    print(f"\n  Processant {method_name.upper()}:")
    print(f"    - Scaler: {best_config['scaler']}")
    print(f"    - Threshold: {best_config['threshold']}")
    print(f"    - K Clusters: {best_config['k_clusters']}")
    print(f"    - Variables: {best_config['vars']}")
    print(f"    - Silhouette: {best_config['silhouette']:.4f}")
    
    # Preparar dades
    scaler = MinMaxScaler() if best_config['scaler'] == 'minmax' else StandardScaler()
    X_scaled_full = scaler.fit_transform(X_original)
    
    # Seleccionar variables segons threshold amb components òptims
    n_pca_comps = optimal_pca_dict.get(best_config['scaler'], 4)
    pca_temp = PCA(n_components=n_pca_comps)
    pca_temp.fit(X_scaled_full)
    loadings_abs = np.abs(pca_temp.components_)
    max_loading = np.max(loadings_abs, axis=0)
    selected_indices = np.where(max_loading >= best_config['threshold'])[0]
    
    X_subset = X_scaled_full[:, selected_indices]
    selected_vars = [column_names[i] for i in selected_indices]
    
    # Aplicar clustering
    labels = apply_clustering(X_subset, method_name, k_clusters=int(best_config['k_clusters']))
    
    # Crear visualitzacions
    plt.figure(figsize=(12, 5))
    
    # PCA
    plt.subplot(1, 2, 1)
    pca_viz = PCA(n_components=2)
    X_pca = pca_viz.fit_transform(X_subset)
    
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', 
                         alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title(f'PCA - {method_name.upper()}\nSilhouette: {best_config["silhouette"]:.4f}')
    plt.grid(True, alpha=0.3)
    
    # t-SNE
    plt.subplot(1, 2, 2)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_subset)
    
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', 
                         alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(f't-SNE - {method_name.upper()}\nK={int(best_config["k_clusters"])}, Vars: {len(selected_vars)}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'viz_best_{method_name}_pca_tsne.png'), dpi=150)
    plt.close()
    
    print(f"    ✓ Visualitzacions guardades")

# ==============================================================================
# 8. EXECUCIÓ PRINCIPAL
# ==============================================================================
print("\n--- 2. Determinant Components Òptims PCA ---")
optimal_components_standard = find_optimal_components(X, 'standard', variance_threshold=0.90)
optimal_components_minmax = find_optimal_components(X, 'minmax', variance_threshold=0.90)

optimal_pca_dict = {
    'standard': optimal_components_standard,
    'minmax': optimal_components_minmax
}

print("\n--- 3. Determinant K Òptim amb Elbow Method ---")
# Preparar dades per trobar K òptim
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)
optimal_k_standard = find_optimal_k(X_std, 'standard', max_k=10)

scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X)
optimal_k_minmax = find_optimal_k(X_mm, 'minmax', max_k=10)

optimal_k_dict = {
    'standard': optimal_k_standard,
    'minmax': optimal_k_minmax
}

print("\n--- 4. Generant Informes PCA ---")
with open(txt_output_path, 'w', encoding='utf-8') as f:
    f.write("INFORME ESTRUCTURAT: PCA & CLUSTERING\n")
    f.write("="*60 + "\n")
    f.write(f"\nComponents òptims determinats (llindar 90%):\n")
    f.write(f"  - Standard Scaler: {optimal_components_standard} components\n")
    f.write(f"  - MinMax Scaler: {optimal_components_minmax} components\n")
    f.write(f"\nK òptim determinat (Elbow Method):\n")
    f.write(f"  - Standard Scaler: {optimal_k_standard} clusters\n")
    f.write(f"  - MinMax Scaler: {optimal_k_minmax} clusters\n")
    
    run_pca_analysis(X, 'standard', optimal_components_standard, cols, f)
    f.write("\n")
    run_pca_analysis(X, 'minmax', optimal_components_minmax, cols, f)

print("\n--- 5. Avaluant Llindars i Silhouette Score ---")
print("Calculant escenaris... (això pot trigar uns segons)")

clustering_methods = ['kmeans', 'gmm', 'hierarchical', 'spectral']
all_results = {}

for method in clustering_methods:
    print(f"\n  Calculant amb {method.upper()}...")
    df_eval = evaluate_thresholds(X, cols, clustering_method=method, 
                                  optimal_k_dict=optimal_k_dict,
                                  optimal_pca_dict=optimal_pca_dict)
    all_results[method] = df_eval
    df_eval.to_csv(os.path.join(output_folder, f'evaluacio_variables_silhouette_{method}.csv'), 
                   index=False)

print(f"\nCàlculs finalitzats. CSVs guardats.")

print("\n--- 6. Generant Gràfics Comparatius ---")
for method in clustering_methods:
    df_eval = all_results[method]
    
    if df_eval.empty:
        print(f"  Warning: No hi ha dades per {method}, saltant gràfic...")
        continue
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    sns.lineplot(data=df_eval, x='threshold', y='silhouette', hue='scaler', 
                 marker='o', linewidth=2.5, palette=['blue', 'red'])
    
    plt.title(f'Qualitat dels Clusters (Silhouette) vs Llindar - {method.upper()}', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.xlabel('Threshold (Tall de PCA Loadings)', fontsize=12)
    
    for i in range(df_eval.shape[0]):
        row = df_eval.iloc[i]
        plt.text(row['threshold'], row['silhouette']+0.005, f"v={int(row['num_vars'])}", 
                 ha='center', size='small', weight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'grafic_comparativa_silhouette_{method}.png'))
    plt.close()
    
    print(f"  Gràfic {method} guardat.")

print("\n--- 7. Visualitzant Millors Resultats amb PCA i t-SNE ---")
for method in clustering_methods:
    df_results = all_results[method]
    best_config = get_best_result(df_results, method)
    
    if best_config is not None:
        visualize_best_clustering(X, cols, best_config, method, optimal_pca_dict)

print("\n" + "="*60)
print("PROCÉS COMPLETAT!")
print("="*60)
print(f"\nResultats guardats a la carpeta: '{output_folder}/'")
print("\nArxius generats:")
print("  - Informes PCA detallats (TXT)")
print("  - Scree plots i Elbow plots (PNG)")
print("  - 5 gràfics de comparativa Silhouette (PNG)")
print("  - 5 visualitzacions PCA+t-SNE dels millors resultats (PNG)")
print("  - CSVs amb avaluacions detallades per cada mètode")
print("="*60)