
# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib

# Mòduls d'sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

# Mòduls externs
from xgboost import XGBClassifier
from yellowbrick.cluster import KElbowVisualizer
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

# Supressió de warnings de Matplotlib/font_manager i configuració
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


# ==============================================================================
# 2. MÒDUL DE PREPROCESSAMENT DE DADES
# ==============================================================================

def load_and_preprocess_data(filename='marketing_campaign.csv'):
    """
    Carrega i preprocessa les dades del dataset de marketing, realitzant 
    Feature Engineering i neteja d'outliers/valors nuls.
    
    Returns:
        tuple: (data, X, cols) 
            - data: DataFrame preprocessat
            - X: Matriu de features numpy escalable
            - cols: Llista de noms de columnes seleccionades
    """
    
    # Càrrega de dades
    try:
        data = pd.read_csv(filename, sep="\t")
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: No s'ha trobat '{filename}'.")
    
    # --- FEATURE ENGINEERING ---
    # 1. Edat
    data['Age'] = 2025 - data['Year_Birth']
    
    # 2. Despesa Total
    data['Total_Spending'] = (
        data['MntWines'] + data['MntFruits'] +
        data['MntMeatProducts'] + data['MntFishProducts'] +
        data['MntSweetProducts'] + data['MntGoldProds']
    )
    
    # 3. Variables Familiars
    partner_status = ['Married', 'Together']
    data['Has_Partner'] = data['Marital_Status'].apply(
        lambda x: 1 if x in partner_status else 0
    )
    data['Family_Size'] = 1 + data['Has_Partner'] + data['Kidhome'] + data['Teenhome']
    
    # 4. Antiguitat Client (Tenure)
    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], dayfirst=True)
    data['Tenure_Days'] = (data['Dt_Customer'].max() - data['Dt_Customer']).dt.days
    
    # --- NETEJA DE DADES I FILTRATGE D'OUTLIERS ---
    # 1. Eliminació de NaNs a 'Income'
    data = data.dropna(subset=['Income'])
    
    # 2. Filtració per valors extrems (Age, Income)
    data = data[(data['Age'] < 100) & (data['Income'] < 600000)]
    invalid_status = ['YOLO', 'Absurd', 'Alone']
    data = data[~data['Marital_Status'].isin(invalid_status)]
    
    # 3. Outliers Spending (IQR * 2.0)
    Q1_spend = data['Total_Spending'].quantile(0.25)
    Q3_spend = data['Total_Spending'].quantile(0.75)
    IQR_spend = Q3_spend - Q1_spend
    data = data[data['Total_Spending'] <= (Q3_spend + 2.0 * IQR_spend)]
    
    # 4. Outliers Income (IQR * 1.5)
    Q1_inc = data['Income'].quantile(0.25)
    Q3_inc = data['Income'].quantile(0.75)
    data = data[data['Income'] <= (Q3_inc + 1.5 * (Q3_inc - Q1_inc))]
    
    # --- PREFERÈNCIES DE PRODUCTE (RATIOS) ---
    epsilon = 1e-6 # Per evitar divisió per zero
    data['Wine_Ratio'] = data['MntWines'] / (data['Total_Spending'] + epsilon)
    data['Meat_Ratio'] = data['MntMeatProducts'] / (data['Total_Spending'] + epsilon)
    data['Sweet_Ratio'] = data['MntSweetProducts'] / (data['Total_Spending'] + epsilon)
    data['Fish_Ratio'] = data['MntFishProducts'] / (data['Total_Spending'] + epsilon)
    data['Fruit_Ratio'] = data['MntFruits'] / (data['Total_Spending'] + epsilon)
    data['Gold_Ratio'] = data['MntGoldProds'] / (data['Total_Spending'] + epsilon)
    
    # --- SELECCIÓ FINAL DE VARIABLES (PER AL CLUSTERING) ---
    selected_columns = [
        # Variables de comportament de compra (absolutes)
        'Income', 'Total_Spending', 
        'MntWines', 'MntMeatProducts', 'MntFishProducts',
        'MntFruits', 'MntSweetProducts', 'MntGoldProds',
        
        # Preferències de producte (ratios normalitzats per spending)
        'Wine_Ratio', 'Meat_Ratio', 'Sweet_Ratio', 
        'Fish_Ratio', 'Fruit_Ratio', 'Gold_Ratio',
        
        # Engagement i context
        'Tenure_Days', 'Family_Size', 'Age'
    ]
    
    X = data[selected_columns].values
    
    return data, X, selected_columns


# ==============================================================================
# 3. UTILITATS COMUNES PER CLUSTERING
# ==============================================================================

def apply_clustering(X_subset, method, k_clusters=4):
    """
    Aplica el mètode de clustering especificat.
    
    Args:
        X_subset (np.array): Matriu de dades
        method (str): Mètode de clustering ('kmeans', 'gmm', 'hierarchical', 'spectral')
        k_clusters (int): Nombre de clusters
    
    Returns:
        np.array: Etiquetes de cluster
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


def calculate_extra_metrics(X_subset, labels):
    """
    Calcula mètriques adicionals de qualitat del clustering basades en la variància:
    - SSE Normalitzat: Cohesió intra-cluster (desitjable: baix)
    - BSS Normalitzat: Separació inter-cluster (desitjable: alt)
    - Correlació Pearson: Relació incidència-proximitat (desitjable: prop de -1)
    
    Args:
        X_subset (np.array): Matriu de dades
        labels (np.array): Etiquetes de cluster
    
    Returns:
        tuple: (sse_norm, bss_norm, corr)
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.nan, np.nan, np.nan
    
    # SSE (Within-cluster Sum of Squares)
    sse = 0
    centers = np.array([X_subset[labels == i].mean(axis=0) for i in unique_labels])
    for i, label in enumerate(unique_labels):
        cluster_points = X_subset[labels == label]
        if len(cluster_points) > 0:
            # Distància quadràtica dels punts al seu propi centre
            sse += np.sum(cdist(cluster_points, [centers[i]])**2)
    
    # SST (Total Sum of Squares)
    global_mean = X_subset.mean(axis=0)
    sst = np.sum((X_subset - global_mean)**2)
    
    # BSS (Between-cluster Sum of Squares)
    bss = sst - sse
    
    # Normalitzar per SST
    sse_norm = sse / sst if sst > 1e-6 else np.nan
    bss_norm = bss / sst if sst > 1e-6 else np.nan
    
    # Correlació Pearson (Incidència vs Distància)
    # Mostreig per a un càlcul més ràpid
    idx = np.random.choice(len(X_subset), min(len(X_subset), 1000), replace=False)
    X_sample = X_subset[idx]
    labels_sample = labels[idx]
    
    if len(X_sample) < 2:
        corr = np.nan
    else:
        # Matriu d'incidència: 1 si els punts estan al mateix cluster, 0 altrament
        incidence_matrix = (labels_sample[:, None] == labels_sample[None, :]).astype(int)
        # Matriu de distàncies
        dist_matrix = pairwise_distances(X_sample)
        
        # Càlcul de la correlació
        try:
            corr, _ = pearsonr(incidence_matrix.flatten(), dist_matrix.flatten())
        except ValueError:
            corr = 0.0 # Valors constants
    
    return sse_norm, bss_norm, corr


def print_and_save_cluster_profile_full(data_df, labels, case_name, out_folder, 
                                        main_txt_path, extra_metrics):
    """
    Imprimeix i guarda un perfil complet dels clústers (mitjanes de variables 
    i recompte), juntament amb mètriques de qualitat.
    """
    tmp = data_df.copy()
    tmp['_CLUSTER_'] = labels
    
    # Perfil: Mitjanes de variables numèriques per clúster
    num_cols = tmp.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['_CLUSTER_']
    num_cols = [c for c in num_cols if c not in exclude_cols]
    summary = tmp.groupby('_CLUSTER_')[num_cols].mean().round(2)
    
    # Recompte de clients per clúster
    counts = tmp['_CLUSTER_'].value_counts().sort_index()
    counts = counts.reindex(summary.index, fill_value=0) 
    summary['Count (Clients)'] = counts.values
    
    sse_n, bss_n, corr = extra_metrics
    
    # Header amb mètriques de qualitat
    metrics_header = (
        f"\n\n\n{'='*90}\n"
        f"📊 PERFIL COMPLET: {case_name}\n"
        f"{'-'*90}\n"
        f"   >> BSS (Separació) Norm: {bss_n:.2%}  (Més alt millor)\n"
        f"   >> SSE (Cohesió) Norm:   {sse_n:.2%}  (Més baix millor)\n"
        f"   >> Correlació (Pearson): {corr:.4f}   (Més a prop de -1 millor)\n"
        f"{'='*90}"
    )
    
    # Guardar CSV i afegir a l'arxiu de text principal
    safe_name = case_name.lower().replace(" ", "_").replace("/", "_").replace("|", "")
    csv_path = f'{out_folder}/perfil_clusters_{safe_name}.csv'
    summary.T.to_csv(csv_path, sep='\t')
    
    with open(main_txt_path, "a", encoding="utf-8") as f:
        f.write(metrics_header + "\n\n")
        f.write(summary.T.to_string())
        f.write("\n")


def save_summary_result(summary_txt_path, test_method, clustering_method, 
                        k_clusters, scaler_name, threshold, num_vars, vars_list, 
                        silhouette, sse_norm, bss_norm, corr):
    """
    Guarda els resultats clau de la millor execució de clustering en un fitxer resum.
    """
    
    content = (
        f"{'='*70}\n"
        f"MÈTODE DE SELECCIÓ: {test_method.upper()}\n"
        f"{'-'*70}\n"
        f"MÈTODE CLUSTERING: {clustering_method.upper()} (k={k_clusters})\n"
        f"CONFIGURACIÓ ÒPTIMA: Scaler={scaler_name.upper()} | Threshold={threshold:.3f} \n"
        f"VARIABLES UTILITZADES ({num_vars}): {', '.join(vars_list)}\n"
        f"{'-'*70}\n"
        f"MÈTRIQUES DE QUALITAT:\n"
        f"   - Silhouette Score:       {silhouette:.4f}\n"
        f"   - BSS Norm (Separació):   {bss_norm:.2%} (Major és millor)\n"
        f"   - SSE Norm (Cohesió):     {sse_norm:.2%} (Menor és millor)\n"
        f"   - Correlació (Pearson):   {corr:.4f} (Prop de -1 és millor)\n"
        f"{'='*70}\n\n"
    )
    
    with open(summary_txt_path, "a", encoding="utf-8") as f:
        f.write(content)


def get_best_result(df_results, method_name, min_vars=2):
    """
    Retorna la millor configuració (màxim silhouette score) del DataFrame de resultats.
    
    Args:
        df_results (pd.DataFrame): DataFrame amb resultats de l'avaluació.
        method_name (str): Nom del mètode (no utilitzat, es manté per compatibilitat).
        min_vars (int): Nombre mínim de variables requerides.
    
    Returns:
        pd.Series: Fila amb la millor configuració o None.
    """
    if df_results.empty:
        return None
    
    df_filtered = df_results[df_results['num_vars'] >= min_vars]
    
    if df_filtered.empty:
        return None
    
    best_row = df_filtered.loc[df_filtered['silhouette'].idxmax()]
    return best_row


def create_visualization_pca_tsne(X_subset, labels, method_name, best_config, 
                                  output_path, selected_vars):
    """
    Crea i guarda visualitzacions 2D del clustering utilitzant reducció de 
    dimensionalitat (PCA i t-SNE) per a interpretació.
    
    Args:
        X_subset (np.array): Dades escalades amb les variables seleccionades.
        labels (np.array): Etiquetes de cluster.
        method_name (str): Nom del mètode de clustering.
        best_config (pd.Series): Configuració òptima.
        output_path (str): Ruta per guardar la imatge.
        selected_vars (list): Llista de variables utilitzades.
    """
    plt.figure(figsize=(12, 5))
    
    # 1. PCA (Reducció Lineal)
    plt.subplot(1, 2, 1)
    n_viz_pca = min(2, X_subset.shape[1])
    pca_viz = PCA(n_components=n_viz_pca)
    X_pca = pca_viz.fit_transform(X_subset)
    
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
    plt.title(f'PCA - {method_name.upper()}\nScore: {best_config.get("silhouette", 0):.4f}')
    plt.grid(True, alpha=0.3)
    
    # 2. t-SNE (Reducció No Lineal)
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
    plt.savefig(output_path, dpi=150)
    plt.close()


# ==============================================================================
# 4. TEST XGBOOST - FEATURE IMPORTANCE (MÈTODE ENVELOPMENT)
# ==============================================================================

def run_xgboost_test(data, X, cols, output_folder):
    """
    Executa l'anàlisi complet de selecció de variables mitjançant
    XGBoost Feature Importance amb pseudo-labeling.
    """
    
    # 1. DETERMINACIÓ K ÒPTIMA (MÈTODE DEL COLZE)
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
    optimal_k = visualizer.elbow_value_ if visualizer.elbow_value_ is not None else 4
    visualizer.show(outpath=os.path.join(output_folder, '00_elbow_kmeans.png'))
    plt.close()
    
    # 2. ANÀLISI D'IMPORTÀNCIA AMB XGBOOST
    def get_xgboost_importance(X_input, column_names, scaler_name, optimal_k):
        """Calcula la importància de features mitjançant XGBoost amb pseudo-labeling."""
        scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
        X_scaled = scaler.fit_transform(X_input)
        
        # Pseudo-labeling: Clusters com a target artificial
        kmeans_base = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        y_pseudo = kmeans_base.fit_predict(X_scaled)
        
        # Entrenar XGBoost com a classificador per predir el pseudo-cluster
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
    
    imp_standard, _ = get_xgboost_importance(X, cols, 'standard', optimal_k)
    imp_minmax, _ = get_xgboost_importance(X, cols, 'minmax', optimal_k)
    
    # Guardar informes i gràfic
    txt_path = os.path.join(output_folder, 'resultats_xgboost_importance.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"INFORME: XGBOOST FEATURE IMPORTANCE (Pseudo-Labeling K={optimal_k})\n")
        f.write("=========================================================\n\n")
        f.write("RÀNQUING (Standard Scaler):\n")
        f.write(imp_standard.to_string(index=False))
        f.write("\n\n")
        f.write("RÀNQUING (MinMax Scaler):\n")
        f.write(imp_minmax.to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_standard, x='Importance', y='Feature', palette='viridis')
    plt.title('XGBoost Feature Importance (Standard Scaler)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'xgboost_feature_importance.png'))
    plt.close()
    
    # 3. AVALUACIÓ CLUSTERING AMB LLINDARS XGBOOST
    def evaluate_xgboost_thresholds(X_original, column_names, imp_df_std, 
                                     imp_df_minmax, k_clusters, clustering_method='kmeans'):
        """Avalua l'Score de Silhouette per diferents llindars d'importància XGBoost."""
        thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
        scalers = ['minmax', 'standard']
        results = []
        
        for scaler_name in scalers:
            current_imp_df = imp_df_minmax if scaler_name == 'minmax' else imp_df_std
            scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
            X_scaled_global = scaler.fit_transform(X_original)
            
            for th in thresholds:
                selected_vars = current_imp_df[current_imp_df['Importance'] >= th]['Feature'].tolist()
                
                if len(selected_vars) < 2:
                    continue
                
                selected_indices = [column_names.index(v) for v in selected_vars]
                X_subset = X_scaled_global[:, selected_indices]
                
                try:
                    labels = apply_clustering(X_subset, clustering_method, k_clusters)
                    
                    if len(set(labels)) >= 2:
                        score = silhouette_score(X_subset, labels)
                        
                        # (Opcional) Penalització per variables no netejades (no rellevant aquí ja netejades al preprocess)
                    else:
                        continue
                        
                    results.append({
                        'scaler': scaler_name,
                        'threshold': th,
                        'num_vars': len(selected_vars),
                        'vars': ", ".join(selected_vars),
                        'silhouette': score
                    })
                except Exception:
                    continue
                
        return pd.DataFrame(results)
    
    clustering_methods = ['kmeans', 'gmm', 'hierarchical', 'spectral']
    all_results = {}
    
    for method in clustering_methods:
        df_eval = evaluate_xgboost_thresholds(
            X, cols, imp_standard, imp_minmax, 
            k_clusters=optimal_k,
            clustering_method=method
        )
        all_results[method] = df_eval
        csv_path = os.path.join(output_folder, f'evaluacio_xgboost_variables_silhouette_{method}.csv')
        df_eval.to_csv(csv_path, index=False)
    
    # 4. GRÀFICS COMPARATIUS
    for method in clustering_methods:
        df_eval = all_results[method]
        
        if df_eval.empty:
            continue
        
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        sns.lineplot(data=df_eval, x='threshold', y='silhouette', hue='scaler', 
                     marker='o', linewidth=2.5, palette=['blue', 'red'])
        
        plt.title(f'Silhouette Score vs XGB Importance Threshold - {method.upper()}', fontsize=14)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.xlabel('XGB Importance Threshold (> X)', fontsize=12)
        
        for i in range(df_eval.shape[0]):
            row = df_eval.iloc[i]
            plt.text(row['threshold'], row['silhouette'], f"v={int(row['num_vars'])}", 
                     ha='center', va='bottom', size='small', weight='bold', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'grafic_comparativa_silhouette_{method}.png'))
        plt.close()
    
    # 5. GENERACIÓ DEL MILLOR RESULTAT (VISUALITZACIÓ + PERFIL)
    def build_labels_and_metrics_for_best_config(X_original, column_names, best_config, 
                                                 method_name, k_clusters):
        """Reprodueix la millor configuració i calcula labels + mètriques"""
        scaler = MinMaxScaler() if best_config['scaler'] == 'minmax' else StandardScaler()
        X_scaled_full = scaler.fit_transform(X_original)
        
        imp_df = imp_minmax if best_config['scaler'] == 'minmax' else imp_standard
        selected_vars = imp_df[imp_df['Importance'] >= best_config['threshold']]['Feature'].tolist()
        selected_indices = [column_names.index(v) for v in selected_vars]
        
        if len(selected_indices) == 0:
            return None, None, None
        
        X_subset = X_scaled_full[:, selected_indices]
        labels = apply_clustering(X_subset, method_name, k_clusters=k_clusters)
        extra_metrics = calculate_extra_metrics(X_subset, labels)
        
        return labels, extra_metrics, selected_vars
    
    txt_output_path = os.path.join(output_folder, 'perfils_clusters_xgboost.txt')
    summary_txt_path = os.path.join(output_folder, 'resum_metriques_best_clusters.txt')
    
    for method in clustering_methods:
        df_results = all_results[method]
        best_config = get_best_result(df_results, method, min_vars=3)
        
        if best_config is None:
            continue
        
        best_labels, extra_metrics, selected_vars = build_labels_and_metrics_for_best_config(
            X_original=X,
            column_names=cols,
            best_config=best_config,
            method_name=method,
            k_clusters=optimal_k
        )
        
        if best_labels is None or extra_metrics is None or np.isnan(extra_metrics[0]):
            continue
        
        # Guardar resum
        save_summary_result(
            summary_txt_path, test_method='XGBoost', clustering_method=method, k_clusters=optimal_k,
            scaler_name=best_config['scaler'], threshold=best_config['threshold'],
            num_vars=len(selected_vars), vars_list=selected_vars, silhouette=best_config['silhouette'],
            sse_norm=extra_metrics[0], bss_norm=extra_metrics[1], corr=extra_metrics[2]
        )
        
        # Visualització
        scaler = MinMaxScaler() if best_config['scaler'] == 'minmax' else StandardScaler()
        X_scaled = scaler.fit_transform(X)
        selected_indices = [cols.index(v) for v in selected_vars]
        X_subset = X_scaled[:, selected_indices]
        
        viz_path = os.path.join(output_folder, f'viz_best_{method}_xgboost_tsne.png')
        create_visualization_pca_tsne(X_subset, best_labels, method, best_config, 
                                      viz_path, selected_vars)
        
        # Perfil detallat
        case_name = (f"XGBOOST BEST | {method.upper()} | "
                    f"scaler={best_config['scaler']} | "
                    f"th={best_config['threshold']} | k={optimal_k}")
        
        print_and_save_cluster_profile_full(data, best_labels, case_name, 
                                           output_folder, txt_output_path, extra_metrics)


# ==============================================================================
# 5. TEST LAPLACIAN - LAPLACIAN SCORE (MÈTODE FILTER)
# ==============================================================================

def run_laplacian_test(data, X, cols, output_folder):
    """
    Executa l'anàlisi complet de selecció de variables mitjançant Laplacian Score.
    """
    
    # 1. DETERMINACIÓ DE K ÒPTIMA (Es reutilitza la visualització de PCA/XGBoost si existeix)
    scaler_elbow = StandardScaler()
    X_scaled_elbow = scaler_elbow.fit_transform(X)
    
    visualizer = KElbowVisualizer(
        KMeans(random_state=42, n_init=10), 
        k=(2, 10), metric='distortion', timings=False
    )
    visualizer.fit(X_scaled_elbow)
    optimal_k = visualizer.elbow_value_ if visualizer.elbow_value_ is not None else 4
    
    # 2. CÀLCUL DE LAPLACIAN SCORE
    def calculate_laplacian_importance(X_input, column_names, scaler_name, k_neighbors=10):
        """Calcula la 'importància' (similitud veïnal) de cada variable."""
        scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
        X_scaled = scaler.fit_transform(X_input)
        
        # Construcció del graf de veïns (Afinity Matrix)
        try:
            affinity_matrix = kneighbors_graph(
                X_scaled, n_neighbors=k_neighbors, mode='connectivity', include_self=False
            ).toarray()
        except Exception:
            return pd.DataFrame({'Feature': column_names, 'Importance': [0.0] * len(column_names)})
        
        # Càlcul del Laplacià i Score
        D = np.diag(np.sum(affinity_matrix, axis=1))
        L = D - affinity_matrix
        
        scores = []
        for i in range(X_scaled.shape[1]):
            feature_vector = X_scaled[:, i].reshape(-1, 1)
            numerator = np.dot(feature_vector.T, np.dot(L, feature_vector))[0, 0]
            denominator = np.dot(feature_vector.T, np.dot(D, feature_vector))[0, 0]
            laplacian_score = numerator / (denominator + 1e-6) # Score baix = variable important
            scores.append(laplacian_score)
        
        # Normalitzar a "Importància" (Invertir el sentit: alt és millor)
        max_score = max(scores)
        min_score = min(scores)
        
        if max_score > min_score:
            importances = [(max_score - s) / (max_score - min_score) for s in scores]
        else:
            importances = [1.0] * len(scores)
        
        df_imp = pd.DataFrame({
            'Feature': column_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        return df_imp
    
    imp_standard = calculate_laplacian_importance(X, cols, 'standard')
    imp_minmax = calculate_laplacian_importance(X, cols, 'minmax')
    
    # Guardar informe i gràfic
    txt_path = os.path.join(output_folder, 'laplacian_importance.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("LAPLACIAN SCORE FEATURE IMPORTANCE\n")
        f.write("=" * 60 + "\n\n")
        f.write("STANDARD SCALER:\n")
        f.write(imp_standard.to_string(index=False))
        f.write("\n\n")
        f.write("MINMAX SCALER:\n")
        f.write(imp_minmax.to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_standard, x='Importance', y='Feature', palette='viridis')
    plt.title('Laplacian Score Feature Importance (Standard Scaler)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'laplacian_importance.png'))
    plt.close()
    
    # 3. AVALUACIÓ DE CLUSTERING AMB DIFERENTS LLINDARS
    def evaluate_thresholds(X_original, column_names, imp_df_std, imp_df_minmax, 
                           k_clusters, method='kmeans'):
        """Avalua l'Score de Silhouette per diferents llindars d'importància Laplacian."""
        thresholds = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
        scalers = ['minmax', 'standard']
        results = []
        
        for scaler_name in scalers:
            current_imp = imp_df_minmax if scaler_name == 'minmax' else imp_df_std
            scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
            X_scaled = scaler.fit_transform(X_original)
            
            for th in thresholds:
                selected_vars = current_imp[current_imp['Importance'] >= th]['Feature'].tolist()
                
                if len(selected_vars) < 2:
                    continue
                
                selected_indices = [column_names.index(v) for v in selected_vars]
                X_subset = X_scaled[:, selected_indices]
                
                try:
                    labels = apply_clustering(X_subset, method, k_clusters)
                    
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
    
    clustering_methods = ['kmeans', 'gmm', 'hierarchical', 'spectral']
    all_results = {}
    
    for method in clustering_methods:
        df_eval = evaluate_thresholds(X, cols, imp_standard, imp_minmax, optimal_k, method)
        all_results[method] = df_eval
        csv_path = os.path.join(output_folder, f'evaluacio_{method}.csv')
        df_eval.to_csv(csv_path, index=False)
    
    # 4. GRÀFICS COMPARATIUS
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
        
        for i in range(df_eval.shape[0]):
            row = df_eval.iloc[i]
            plt.text(row['threshold'], row['silhouette'], f"v={int(row['num_vars'])}", 
                     ha='center', va='bottom', size='small', weight='bold', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'comparativa_{method}.png'))
        plt.close()
    
    # 5. GENERACIÓ DEL MILLOR RESULTAT (VISUALITZACIÓ + PERFIL)
    def build_labels_and_metrics_for_best_config(X_original, column_names, best_config, 
                                                 method_name, k_clusters):
        """Reprodueix la millor configuració i calcula labels + mètriques"""
        if best_config is None:
            return None, None, None
            
        scaler = MinMaxScaler() if best_config['scaler'] == 'minmax' else StandardScaler()
        X_scaled_full = scaler.fit_transform(X_original)
        
        imp_df = imp_minmax if best_config['scaler'] == 'minmax' else imp_standard
        selected_vars = imp_df[imp_df['Importance'] >= best_config['threshold']]['Feature'].tolist()
        selected_indices = [column_names.index(v) for v in selected_vars]
        
        if len(selected_indices) == 0:
            return None, None, None
        
        X_subset = X_scaled_full[:, selected_indices]
        labels = apply_clustering(X_subset, method_name, k_clusters=k_clusters)
        extra_metrics = calculate_extra_metrics(X_subset, labels)
        
        return labels, extra_metrics, selected_vars
    
    txt_output_path = os.path.join(output_folder, 'perfils_clusters_laplacian.txt')
    summary_txt_path = os.path.join(output_folder, 'resum_metriques_best_clusters.txt')
    
    for method in clustering_methods:
        best_config = get_best_result(all_results[method], method, min_vars=3)
        
        if best_config is None:
            continue
        
        best_labels, extra_metrics, selected_vars = build_labels_and_metrics_for_best_config(
            X_original=X, column_names=cols, best_config=best_config, 
            method_name=method, k_clusters=optimal_k
        )
        
        if best_labels is None or extra_metrics is None or np.isnan(extra_metrics[0]):
            continue
        
        # Guardar resum
        save_summary_result(
            summary_txt_path, test_method='Laplacian Score', clustering_method=method, k_clusters=optimal_k,
            scaler_name=best_config['scaler'], threshold=best_config['threshold'],
            num_vars=len(selected_vars), vars_list=selected_vars, silhouette=best_config['silhouette'],
            sse_norm=extra_metrics[0], bss_norm=extra_metrics[1], corr=extra_metrics[2]
        )
        
        # Visualització
        scaler = MinMaxScaler() if best_config['scaler'] == 'minmax' else StandardScaler()
        X_scaled = scaler.fit_transform(X)
        selected_indices = [cols.index(v) for v in selected_vars]
        X_subset = X_scaled[:, selected_indices]
        
        viz_path = os.path.join(output_folder, f'viz_best_{method}.png')
        create_visualization_pca_tsne(X_subset, best_labels, method, best_config, 
                                      viz_path, selected_vars)
        
        # Perfil detallat
        case_name = (f"LAPLACIAN BEST | {method.upper()} | "
                    f"scaler={best_config['scaler']} | "
                    f"th={best_config['threshold']} | k={optimal_k}")
        
        print_and_save_cluster_profile_full(data, best_labels, case_name, 
                                           output_folder, txt_output_path, extra_metrics)


# ==============================================================================
# 6. TEST PCA - PRINCIPAL COMPONENT ANALYSIS (MÈTODE EMBEDDED/WRAPPING)
# ==============================================================================

def run_pca_test(data, X, cols, output_folder):
    """
    Executa l'anàlisi complet de selecció de variables mitjançant PCA Loadings.
    """
    
    txt_output_path = os.path.join(output_folder, 'resultats_pca_clustering_detallats.txt')
    
    # 1. DETERMINANT COMPONENTS ÒPTIMS PCA
    def find_optimal_components(X_input, scaler_name, variance_threshold=0.90):
        """Determina el nombre mínim de components PCA per superar el llindar de variància."""
        scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
        X_scaled = scaler.fit_transform(X_input)
        pca_full = PCA().fit(X_scaled)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        return np.argmax(cumulative_variance >= variance_threshold) + 1
        
    optimal_components_standard = find_optimal_components(X, 'standard', variance_threshold=0.90)
    optimal_components_minmax = find_optimal_components(X, 'minmax', variance_threshold=0.90)
    
    optimal_pca_dict = {
        'standard': optimal_components_standard,
        'minmax': optimal_components_minmax
    }
    
    # 2. DETERMINANT K ÒPTIM AMB ELBOW METHOD
    def find_optimal_k(X_input, scaler_name, output_folder, max_k=10):
        """Determina el nombre òptim de clusters utilitzant el mètode Elbow."""
        model = KMeans(random_state=42, n_init=10)
        visualizer = KElbowVisualizer(model, k=(2, max_k+1), timings=False)
        visualizer.fit(X_input)
        optimal_k = visualizer.elbow_value_
        visualizer.fig.savefig(os.path.join(output_folder, f'elbow_plot_{scaler_name}.png'))
        plt.close()
        return optimal_k if optimal_k is not None else 4
        
    scaler_std = StandardScaler()
    X_std = scaler_std.fit_transform(X)
    optimal_k_standard = find_optimal_k(X_std, 'standard', output_folder, max_k=10)
    
    scaler_mm = MinMaxScaler()
    X_mm = scaler_mm.fit_transform(X)
    optimal_k_minmax = find_optimal_k(X_mm, 'minmax', output_folder, max_k=10)
    
    optimal_k_dict = {
        'standard': optimal_k_standard,
        'minmax': optimal_k_minmax
    }
    
    # 3. GENERANT INFORMES PCA I SCREE PLOTS
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        f.write("INFORME ESTRUCTURAT: PCA & CLUSTERING\n")
        f.write("="*60 + "\n")
        f.write(f"\nComponents òptims determinats (llindar 90%):\n")
        f.write(f"  - Standard Scaler: {optimal_components_standard} components\n")
        f.write(f"  - MinMax Scaler: {optimal_components_minmax} components\n")
        f.write(f"\nK òptim determinat (Elbow Method):\n")
        f.write(f"  - Standard Scaler: {optimal_k_standard} clusters\n")
        f.write(f"  - MinMax Scaler: {optimal_k_minmax} clusters\n")
        
        def run_pca_analysis(X_input, scaler_name, n_components, column_names, 
                            file_handle, output_folder):
            """Executa anàlisi PCA amb scree plot i loadings."""
            title = f"ANÀLISI PCA: {scaler_name.upper()}"
            file_handle.write(f"\n{'#'*60}\n{title}\n{'#'*60}\n")
            
            scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
            X_scaled = scaler.fit_transform(X_input)
            
            # Scree Plot
            pca_full = PCA().fit(X_scaled)
            var = pca_full.explained_variance_ratio_ * 100
            cumulative_var = np.cumsum(var)
            
            plt.figure(figsize=(10,6))
            plt.plot(range(1, len(var)+1), var, marker='o', color='black', label='Variància Individual')
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
            
            # PCA Final i Loadings
            n_comps = min(n_components, X_scaled.shape[1])
            pca = PCA(n_components=n_comps).fit(X_scaled)
            
            file_handle.write(f"\n[2] Varianïa Acumulada ({n_comps} comps): {pca.explained_variance_ratio_.cumsum()[-1]*100:.2f}%\n")
            
            comps_df = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(n_comps)], index=column_names)
            file_handle.write("\n[3] PCA Loadings (Variables més importants per component):\n")
            for i in range(n_comps):
                pc = f"PC{i+1}"
                top = comps_df[pc].abs().sort_values(ascending=False)
                file_handle.write(f"\n>> {pc} ({pca.explained_variance_ratio_[i]*100:.2f}%):\n")
                for vname, val in top.items():
                    real_val = comps_df.loc[vname, pc]
                    file_handle.write(f"  - {vname:20} : {real_val:+.4f}\n")
        
        run_pca_analysis(X, 'standard', optimal_components_standard, cols, f, output_folder)
        f.write("\n")
        run_pca_analysis(X, 'minmax', optimal_components_minmax, cols, f, output_folder)
    
    # 4. AVALUANT LLINDARS DE LOADINGS I SILHOUETTE SCORE
    def evaluate_thresholds(X_original, column_names, clustering_method='kmeans', 
                            optimal_k_dict=None, optimal_pca_dict=None):
        """Avalua diferents llindars de selecció de variables basats en PCA Loadings."""
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        scalers = ['minmax', 'standard']
        results = []
        
        for scaler_name in scalers:
            k_clusters = optimal_k_dict.get(scaler_name, 4)
            n_pca_comps = optimal_pca_dict.get(scaler_name, 4)
            
            scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
            X_scaled_global = scaler.fit_transform(X_original)
            
            # Calcular Loadings màxims per variable
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
                    labels = apply_clustering(X_subset, clustering_method, k_clusters)
                    
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
                except Exception:
                    continue
            
        return pd.DataFrame(results)
    
    clustering_methods = ['kmeans', 'gmm', 'hierarchical', 'spectral']
    all_results = {}
    
    for method in clustering_methods:
        df_eval = evaluate_thresholds(X, cols, clustering_method=method, 
                                      optimal_k_dict=optimal_k_dict,
                                      optimal_pca_dict=optimal_pca_dict)
        all_results[method] = df_eval
        df_eval.to_csv(os.path.join(output_folder, f'evaluacio_variables_silhouette_{method}.csv'), 
                       index=False)
    
    # 5. GENERANT GRÀFICS COMPARATIUS
    for method in clustering_methods:
        df_eval = all_results[method]
        
        if df_eval.empty:
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
    
    # 6. VISUALITZACIÓ MILLORS RESULTATS I PERFIL COMPLET
    
    def build_labels_and_metrics_for_best_config(X_original, column_names, best_config, 
                                                 method_name, optimal_pca_dict):
        """Reprodueix la millor configuració, calcula labels i mètriques extra."""
        if best_config is None:
            return None, None, None
            
        scaler = MinMaxScaler() if best_config['scaler'] == 'minmax' else StandardScaler()
        X_scaled_full = scaler.fit_transform(X_original)
        
        n_pca_comps = optimal_pca_dict.get(best_config['scaler'], 4)
        pca_temp = PCA(n_components=n_pca_comps)
        pca_temp.fit(X_scaled_full)
        loadings_abs = np.abs(pca_temp.components_)
        max_loading = np.max(loadings_abs, axis=0)
        selected_indices = np.where(max_loading >= best_config['threshold'])[0]
        
        selected_vars = [column_names[i] for i in selected_indices]
        
        if len(selected_indices) == 0:
            return None, None, None
        
        X_subset = X_scaled_full[:, selected_indices]
        labels = apply_clustering(X_subset, method_name, k_clusters=int(best_config['k_clusters']))
        extra_metrics = calculate_extra_metrics(X_subset, labels)
        
        return labels, extra_metrics, selected_vars
        
    summary_txt_path = os.path.join(output_folder, 'resum_metriques_best_clusters.txt')
    
    for method in clustering_methods:
        df_results = all_results[method]
        best_config = get_best_result(df_results, method, min_vars=3) 
        
        if best_config is None:
            continue
        
        best_labels, extra_metrics, selected_vars = build_labels_and_metrics_for_best_config(
            X_original=X, column_names=cols, best_config=best_config, 
            method_name=method, optimal_pca_dict=optimal_pca_dict
        )
        
        if best_labels is None or extra_metrics is None or np.isnan(extra_metrics[0]):
            continue
        
        # Guardar resum de mètriques
        save_summary_result(
            summary_txt_path, test_method='PCA Loadings', clustering_method=method, 
            k_clusters=int(best_config['k_clusters']), scaler_name=best_config['scaler'], 
            threshold=best_config['threshold'], num_vars=len(selected_vars), 
            vars_list=selected_vars, silhouette=best_config['silhouette'],
            sse_norm=extra_metrics[0], bss_norm=extra_metrics[1], corr=extra_metrics[2]
        )
        
        # Visualització (Reutilitzant la funció comuna)
        scaler = MinMaxScaler() if best_config['scaler'] == 'minmax' else StandardScaler()
        X_scaled = scaler.fit_transform(X)
        selected_indices = [cols.index(v) for v in selected_vars]
        X_subset = X_scaled[:, selected_indices]
        
        viz_path = os.path.join(output_folder, f'viz_best_{method}_pca_tsne.png')
        create_visualization_pca_tsne(X_subset, best_labels, method, best_config, 
                                      viz_path, selected_vars)
        
        # Perfil detallat
        case_name = (f"PCA BEST | {method.upper()} | "
                    f"scaler={best_config['scaler']} | "
                    f"th={best_config['threshold']} | k={int(best_config['k_clusters'])}")
        
        print_and_save_cluster_profile_full(data, best_labels, case_name, 
                                           output_folder, txt_output_path, extra_metrics)


# ==============================================================================
# 7. PIPELINE PRINCIPAL D'EXECUCIÓ
# ==============================================================================

if __name__ == "__main__":
    # CONFIGURACIÓ INICIAL
    MAIN_FOLDER = 'resultats_test'
    OUTPUT_FOLDERS = {
        'pca': os.path.join(MAIN_FOLDER, 'test_PCA'),
        'xgboost': os.path.join(MAIN_FOLDER, 'test_XGBoost'),
        'laplacian': os.path.join(MAIN_FOLDER, 'test_Laplacian')
    }

    # Netejar i crear carpetes de resultats
    if os.path.exists(MAIN_FOLDER):
        shutil.rmtree(MAIN_FOLDER)

    for folder in OUTPUT_FOLDERS.values():
        os.makedirs(folder)

    # 1. Càrrega i preprocessament de dades (comú per tots)
    try:
        data, X, cols = load_and_preprocess_data()
        print(f"✅ Dades processades correctament: {len(data)} registres")
    except Exception as e:
        print(f"❌ Error en la càrrega/preprocessament: {e}")
        exit()
    
    # 2. Test PCA (Anàlisi de Loadings com a Feature Selection)
    try:
        run_pca_test(data, X, cols, OUTPUT_FOLDERS['pca'])
        print("✅ Test PCA completat correctament")
    except Exception as e:
        print(f"❌ Error en Test PCA: {e}")
    
    # 3. Test XGBoost (Feature Importance amb Pseudo-Labeling)
    try:
        run_xgboost_test(data, X, cols, OUTPUT_FOLDERS['xgboost'])
        print("✅ Test XGBoost completat correctament")
    except Exception as e:
        print(f"❌ Error en Test XGBoost: {e}")
    
    # 4. Test Laplacian Score (Mètrica de similitud veïnal)
    try:
        run_laplacian_test(data, X, cols, OUTPUT_FOLDERS['laplacian'])
        print("✅ Test Laplacian completat correctament")
    except Exception as e:
        print(f"❌ Error en Test Laplacian: {e}")
    
    # Tancament
    print("\n" + "="*60)
    print("TOTS ELS TESTS COMPLETATS!")
    print("="*60)
    print(f"\nResultats guardats a: '{MAIN_FOLDER}/'")
    print(f"Els resums de mètriques finals es troben a 'resum_metriques_best_clusters.txt' dins de cada subcarpeta.")
    print("="*60)