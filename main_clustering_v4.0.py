# ==============================================================================
# MASTER CLUSTERING SCRIPT - DUAL EXECUTION (XGBOOST & PCA) - BALANCED SCORE
# ==============================================================================
import os
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn & Stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr
from xgboost import XGBClassifier, plot_importance
from yellowbrick.cluster import KElbowVisualizer

warnings.filterwarnings('ignore')

# 0. CONFIGURACIÓ
output_folder = 'resultats_clustering_dual'
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)
print(f"Carpeta '{output_folder}' preparada.")

## ==============================================================================
# 1. DATA LOADING & CLEANING (ADVANCED: RATIOS + POWER TRANSFORMER)
# ==============================================================================
# Afegim l'import necessari aquí dalt per si no el tens
from sklearn.preprocessing import PowerTransformer

print("\n--- 1. Càrrega i Preprocessament Avançat ---")
filename = 'marketing_campaign.csv' # O el nom que tinguis

try:
    data = pd.read_csv(filename, sep="\t")
except FileNotFoundError:
    print("Error: No s'ha trobat el fitxer CSV.")
    exit()

# 1.1 Feature Engineering Bàsic
data['Age'] = 2025 - data['Year_Birth']
data = data[data['Age'] < 80]

# 1.2 Càlcul de Despesa Total
product_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
data['Total_Spending'] = data[product_cols].sum(axis=1)

# FILTRE DE SEGURETAT: Eliminem gent que no gasta res (distorsionen els ràtios)
data = data[data['Total_Spending'] > 5]

# 1.3 CREACIÓ DE RÀTIOS (La Millora Clau: Perfils de Gust)
# Això permet diferenciar "El que compra Vi" del "El que compra Carn",
# independentment de si és ric o pobre.
for col in product_cols:
# Exemple: Pct_Wines serà el % del pressupost destinat a vi
    data[f'Pct_{col}'] = data[col] / data['Total_Spending']

# 1.4 Feature Engineering: Família i Temps
partner_status = ['Married', 'Together']
data['Has_Partner'] = data['Marital_Status'].apply(lambda x: 1 if x in partner_status else 0)
data['Family_Size'] = 1 + data['Has_Partner'] + data['Kidhome'] + data['Teenhome']

data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], dayfirst=True)
data['Tenure_Days'] = (data['Dt_Customer'].max() - data['Dt_Customer']).dt.days

# Mapejos Categòrics
education_map = {'Basic': 1, '2n Cycle': 2, 'Graduation': 3, 'Master': 4, 'PhD': 5}
data['Education_Code'] = data['Education'].map(education_map).fillna(0)

marital_map = {'Married': 1, 'Together': 2, 'Divorced': 3, 'Widow': 3, 'Single': 3}
data['Marital_Status_Code'] = data['Marital_Status'].map(marital_map).fillna(0)

# Neteja de Nuls i Errors d'entrada
data = data.dropna(subset=['Income', 'Education_Code'])
data = data[data['Income'] < 600000] # Eliminem error de teclat típic (666.666)
data = data[~data['Marital_Status'].isin(['YOLO', 'Absurd', 'Alone'])]

# 1.5 TRANSFORMACIÓ GAUSSIANA (PowerTransformer)
# En lloc de fer Log manual i Outliers IQR manual, això ho fa tot automàticament.
# Converteix les distribucions lletges en campanes de Gauss perfectes per al K-Means.

cols_to_transform = [
'Income', 'Total_Spending', 'Age', 'Tenure_Days', 'Recency',
# Afegim també els productes bruts per si els vols fer servir
'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]

# Inicialitzem el transformador (Yeo-Johnson funciona amb positius i negatius/zeros)
pt = PowerTransformer(method='yeo-johnson')

# Apliquem la transformació
data[cols_to_transform] = pt.fit_transform(data[cols_to_transform])

# NOTA: Els 'Pct_' (Ràtios) ja estan entre 0 i 1, normalment no cal tocar-los molt,
# però si vols ser purista pots transformar-los també. De moment els deixem naturals.

print(f"Dades processades (Ràtios + PowerTransformer): {len(data)} clients.")


# ==============================================================================
# 2. SELECCIÓ DE VARIABLES (DUAL)
# ==============================================================================
print("\n--- 2. Selecció de Variables: XGBoost vs PCA ---")

# Llista de candidates (sense Total_Spending si usem els individuals, o al revés)
candidate_cols = [
    'Income', 'Family_Size', 'Seniority_Code', 'Age',
    'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'Education_Code', 'Marital_Status_Code',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]
candidate_cols = [c for c in candidate_cols if c in data.columns]

# Funció auxiliar de mètriques i score balançat
def calculate_metrics_score(X_subset):
    if X_subset.shape[1] < 1: return 0, 0, 0
    k=4 # K fixa per comparar ràpid
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_subset)
    labels = model.labels_; centers = model.cluster_centers_
    
    sse = 0
    for i in range(k):
        mask = (labels == i)
        if np.any(mask): sse += np.sum(cdist(X_subset[mask], [centers[i]])**2)
    mse = sse / len(X_subset)
    bss = np.sum(pdist(centers))
    
    idx = np.random.choice(len(X_subset), min(len(X_subset), 1000), replace=False)
    inc = (labels[idx,None] == labels[None,idx]).astype(int)
    prox = pairwise_distances(X_subset[idx])
    corr, _ = pearsonr(inc.flatten(), prox.flatten())
    
    return mse, bss, corr

def get_normalized_score(mse_list, bss_list, corr_list):
    def norm(arr):
        if np.max(arr) == np.min(arr): return np.zeros_like(arr)
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    
    s_bss = norm(np.array(bss_list))       # Més alt millor
    s_mse = norm(-1 * np.array(mse_list))  # Més baix millor (invertim)
    s_corr = norm(-1 * np.array(corr_list))# Més baix millor (invertim)
    
    return (0.33 * s_bss) + (0.33 * s_mse) + (0.33 * s_corr)

# --- BRANCA A: XGBOOST ---
print("\n🔹 BRANCA A: Selecció via XGBoost")
X_all = np.nan_to_num(StandardScaler().fit_transform(data[candidate_cols].values))
kmeans_base = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_all)
clf = XGBClassifier(n_estimators=100, random_state=42).fit(X_all, kmeans_base.labels_)
sorted_features_xgb = [candidate_cols[i] for i in np.argsort(clf.feature_importances_)[::-1]]
print(f"   Rànquing XGB: {sorted_features_xgb}")

hist_xgb = {'n': [], 'mse': [], 'bss': [], 'corr': [], 'vars': []}

for i in range(2, len(sorted_features_xgb) + 1):
    cols = sorted_features_xgb[:i]
    X_sub = np.nan_to_num(StandardScaler().fit_transform(data[cols].values))
    mse, bss, corr = calculate_metrics_score(X_sub)
    
    hist_xgb['n'].append(i); hist_xgb['mse'].append(mse); hist_xgb['bss'].append(bss)
    hist_xgb['corr'].append(corr); hist_xgb['vars'].append(cols)

xgb_scores = get_normalized_score(hist_xgb['mse'], hist_xgb['bss'], hist_xgb['corr'])
best_idx_xgb = np.argmax(xgb_scores)
vars_xgb = hist_xgb['vars'][best_idx_xgb]

plt.figure(figsize=(8, 4))
plt.plot(hist_xgb['n'], xgb_scores, 'o-', color='purple')
plt.title('XGBoost: Score Balançat (BSS/MSE/Corr)')
plt.savefig(f'{output_folder}/01a_xgboost_selection.png'); plt.close()
print(f"   -> Millor XGBoost: {len(vars_xgb)} variables.")


# --- BRANCA B: PCA ---
print("\n🔸 BRANCA B: Selecció via PCA (Automàtica >80%)")

# 1. Ajustem PCA
pca = PCA().fit(X_all)

# 2. Calculem quants components calen per explicar el 80%
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_optimal = np.argmax(cumulative_variance >= 0.80) + 1 # +1 perquè l'índex comença a 0

print(f"   -> Per explicar el 80% de la variància calen: {n_components_optimal} components.")
print(f"      (Variància real acumulada: {cumulative_variance[n_components_optimal-1]:.2%})")

# 3. GRÀFIC DE VARIÀNCIA (SCREE PLOT)
plt.figure(figsize=(8, 5))
# Barres: Variància individual de cada component
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5, align='center', label='Var. Individual')
# Línia: Variància acumulada
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Var. Acumulada', color='red')
# Línies de referència
plt.axhline(y=0.80, color='k', linestyle='--', label='Llindar 80%')
plt.axvline(x=n_components_optimal, color='green', linestyle=':', label=f'Tall Òptim ({n_components_optimal})')

plt.ylabel('Ràtio de Variància Explicada')
plt.xlabel('Components Principals')
plt.title(f'PCA Scree Plot (Tall al 80%: {n_components_optimal} Comps)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_folder}/01b_pca_variance.png')
plt.close() # Neteja memòria

# 4. Anàlisi de Pesos (Loadings)
loadings = pd.DataFrame(pca.components_.T, index=candidate_cols)

print(f"\n   Pesos dels {n_components_optimal} components principals:")
# Mostrem només els components rellevants
print(loadings.iloc[:, :n_components_optimal].round(3)) 

# 5. Bucle de Selecció (Thresholding)
hist_pca = {'thresh': [], 'mse': [], 'bss': [], 'corr': [], 'vars': []}
thresholds = [0.8, 0.7, 0.6, 0.5, 0.4]

print("\n   Analitzant llindars...")
for thresh in thresholds:
    important_vars = set()
    
    # ARA EL BUCLE ÉS DINÀMIC: Només mira els components necessaris (n_components_optimal)
    for i in range(n_components_optimal): 
        if i < loadings.shape[1]:
            # Usem .iloc per seguretat
            vars_in_pc = loadings.index[loadings.iloc[:, i].abs() > thresh].tolist()
            important_vars.update(vars_in_pc)
            
    current_vars = list(important_vars)
    
    if len(current_vars) < 3:
        hist_pca['thresh'].append(str(thresh))
        hist_pca['mse'].append(999); hist_pca['bss'].append(0); hist_pca['corr'].append(1); hist_pca['vars'].append([])
        continue
        
    X_sub = np.nan_to_num(StandardScaler().fit_transform(data[current_vars].values))
    mse, bss, corr = calculate_metrics_score(X_sub)
    
    hist_pca['thresh'].append(str(thresh))
    hist_pca['mse'].append(mse); hist_pca['bss'].append(bss); hist_pca['corr'].append(corr); hist_pca['vars'].append(current_vars)

pca_scores = get_normalized_score(hist_pca['mse'], hist_pca['bss'], hist_pca['corr'])
best_idx_pca = np.argmax(pca_scores)
vars_pca = hist_pca['vars'][best_idx_pca]

# Gràfic de Selecció PCA
plt.figure(figsize=(8, 4))
plt.bar(hist_pca['thresh'], pca_scores, color='orange')
plt.title('PCA: Score Balançat segons Llindar')
plt.xlabel('Llindar de Càrrega'); plt.ylabel('Score Normalitzat')
# Posem el nombre de variables a les barres
for i, v in enumerate(pca_scores):
    if v > 0: plt.text(i, v, str(len(hist_pca['vars'][i])), ha='center', va='bottom')
plt.savefig(f'{output_folder}/01b_pca_selection.png'); plt.close()

print(f"   -> Millor PCA: {len(vars_pca)} variables.")
# ==============================================================================
# 3. DOBLE EXECUCIÓ FINAL (XGBOOST I PCA PER SEPARAT)
# ==============================================================================

def run_full_clustering(variable_set, method_name):
    if len(variable_set) < 2:
        print(f"⚠️ {method_name} no té prous variables per executar clustering.")
        return

    print(f"\n{'='*60}")
    print(f"🚀 EXECUTANT CLUSTERING COMPLET PER: {method_name.upper()}")
    print(f"   Variables ({len(variable_set)}): {variable_set}")
    print(f"{'='*60}")
    
    # 1. Preparar Dades
    X_final = data[variable_set].values
    X_scaled = np.nan_to_num(StandardScaler().fit_transform(X_final))
    
    # 2. K-Òptima
    plt.close('all'); plt.figure(figsize=(8, 5))
    viz = KElbowVisualizer(KMeans(random_state=42, n_init=10), k=(2,10), timings=False)
    viz.fit(X_scaled)
    viz.show(outpath=f"{output_folder}/02_elbow_{method_name}.png"); plt.close()
    k_opt = viz.elbow_value_
    print(f"   K Òptima ({method_name}): {k_opt}")
    
    # 3. Models
    models = {
        'KMeans': KMeans(n_clusters=k_opt, random_state=42),
        'Jerarquic': AgglomerativeClustering(n_clusters=k_opt, linkage='ward'),
        'GMM': GaussianMixture(n_components=k_opt, random_state=42),
        'Spectral': SpectralClustering(n_clusters=k_opt, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42, n_jobs=-1)
    }
    
    # TSS per normalitzar (BSS %)
    global_mean = X_scaled.mean(axis=0)
    tss = np.sum((X_scaled - global_mean) ** 2)
    
    for model_type, model in models.items():
        print(f"   Running {model_type}...")
        labels = model.fit_predict(X_scaled)
        col_name = f'{method_name}_{model_type}'
        data[col_name] = labels # Guardem al df general
        
        # Plot
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        comp = tsne.fit_transform(X_scaled)
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=comp[:,0], y=comp[:,1], hue=labels, palette='viridis', s=50)
        plt.title(f'{method_name} - {model_type} (K={k_opt})')
        plt.savefig(f'{output_folder}/03_plot_{method_name}_{model_type}.png'); plt.close()
        
        # Mètriques (AMB CORRELACIÓ AFEGIDA)
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(X_scaled, labels)
            
            centers = np.array([X_scaled[labels==i].mean(axis=0) for i in np.unique(labels)])
            sse = 0; bss_stat = 0
            for i in np.unique(labels):
                mask = (labels == i); points = X_scaled[mask]; n = len(points)
                if n > 0:
                    sse += np.sum(cdist(points, [centers[i]])**2)
                    bss_stat += n * np.sum((centers[i] - global_mean)**2)
            
            # Càlcul Correlació
            idx = np.random.choice(len(X_scaled), min(len(X_scaled), 1000), replace=False)
            inc = (labels[idx,None] == labels[None,idx]).astype(int)
            prox = pairwise_distances(X_scaled[idx])
            corr, _ = pearsonr(inc.flatten(), prox.flatten())

            print(f"      📊 {model_type}: Sil={sil:.3f} | BSS%={bss_stat/tss:.1%} | SSE%={sse/tss:.1%} | Corr={corr:.4f}")

    # 4. Validació XGBoost (Només per KMeans)
    print(f"   Validant KMeans amb XGBoost...")
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, data[f'{method_name}_KMeans'], test_size=0.2, random_state=42)
    val_clf = XGBClassifier(eval_metric='mlogloss').fit(X_tr, y_tr)
    acc = val_clf.score(X_te, y_te)
    print(f"   ✅ Accuracy Classificació: {acc:.2%}")
    
    # 5. Profiling (Només KMeans)
    cols_prof = variable_set + ['Age']
    summary = data.groupby(f'{method_name}_KMeans')[cols_prof].mean().T
    # Invertim logs per llegir diners reals
    for c in summary.index:
        if 'Mnt' in c or 'Total' in c or 'Income' in c:
            summary.loc[c] = np.expm1(summary.loc[c])
    
    print(f"\n   Perfil {method_name} (KMeans - Mitjanes):")
    print(summary.round(2))


# --- EXECUCIÓ DOBLE ---
run_full_clustering(vars_xgb, 'XGBoost_Vars')
run_full_clustering(vars_pca, 'PCA_Vars')

# Guardar
data.to_csv(f'{output_folder}/final_dual_results.csv', sep='\t', index=False)
print("\nProcés Dual Finalitzat! 🚀")