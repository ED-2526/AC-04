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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import xgboost as xgb

# ==============================================================================
# 0. CONFIGURACIÓ INICIAL
# ==============================================================================
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

output_folder = 'resultats_clustering'
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

print(f"Carpeta '{output_folder}' preparada.")

txt_output_path = os.path.join(output_folder, 'resultats_pca_xgboost_detallats.txt')

selected_columns = [
    "Age", "Total_Spending", "Income", "Family_Size", 
    "Seniority_Code", "Education_Code", "Marital_Status_Code"
]

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

print(f"Dades netes: {len(data)} registres.")

# Target per XGBoost
if 'Response' in data.columns:
    y_target = data['Response'].values
    target_name = "Response (Campaign Acceptance)"
else:
    y_target = np.random.randint(0, 2, size=len(data))
    target_name = "Dummy Target"

X = data[selected_columns].values
cols = selected_columns

# ==============================================================================
# 2. FUNCIÓ D'ANÀLISI PCA
# ==============================================================================
def run_pca_analysis(X_input, scaler_name, n_components, column_names, file_handle):
    title = f"ANÀLISI PCA: {scaler_name.upper()}"
    file_handle.write(f"\n{'#'*60}\n{title}\n{'#'*60}\n")
    
    scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
    X_scaled = scaler.fit_transform(X_input)
    
    file_handle.write("\n[1] Desviació Estàndard:\n")
    std_s = pd.Series(X_scaled.std(axis=0), index=column_names).sort_values(ascending=False)
    for k, v in std_s.items(): file_handle.write(f"{k:20}: {v:.4f}\n")

    pca_full = PCA().fit(X_scaled)
    var = pca_full.explained_variance_ratio_ * 100
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(var)+1), var, marker='o', color='black')
    plt.bar(range(1, len(var)+1), var, alpha=0.7)
    plt.title(f'Scree Plot ({scaler_name})')
    plt.savefig(os.path.join(output_folder, f'scree_plot_{scaler_name}.png'))
    plt.close()

    n_comps = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=n_comps).fit(X_scaled)
    
    file_handle.write(f"\n[2] Variança Acumulada ({n_comps} comps): {pca.explained_variance_ratio_.cumsum()[-1]*100:.2f}%\n")
    
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
# 3. FUNCIÓ D'ANÀLISI XGBOOST
# ==============================================================================
def run_xgboost_analysis(X_raw, y, column_names, file_handle):
    print("Executant anàlisi XGBoost...")
    file_handle.write(f"\n{'#'*60}\nANÀLISI XGBOOST (IMPORTÀNCIA PREDICTIVA)\n{'#'*60}\n")
    file_handle.write(f"Target utilitzat: {target_name}\n\n")

    results = {}
    scalers = {'minmax': MinMaxScaler(), 'standard': StandardScaler()}
    
    for name, scaler in scalers.items():
        X_sc = scaler.fit_transform(X_raw)
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, 
                                  use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_sc, y)
        imp_series = pd.Series(model.feature_importances_, index=column_names)
        results[name] = imp_series
        
        file_handle.write(f"[{name.upper()}] Rànquing Importància:\n")
        for var, val in imp_series.sort_values(ascending=False).items():
            file_handle.write(f"   - {var:20} : {val:.4f}\n")
        file_handle.write("\n")
        
    return pd.DataFrame(results)

# ==============================================================================
# 4. EXECUCIÓ (Generació d'Informes)
# ==============================================================================
print("\n--- 2. Generant informes PCA i XGBoost ---")

with open(txt_output_path, 'w', encoding='utf-8') as f:
    f.write("INFORME ESTRUCTURAT: PCA & XGBOOST\n")
    
    run_pca_analysis(X, 'standard', 6, cols, f)
    f.write("\n")
    run_pca_analysis(X, 'minmax', 5, cols, f)
    
    df_xgb = run_xgboost_analysis(X, y_target, cols, f)

# Gràfic Comparatiu XGBoost
print("Generant gràfic comparatiu XGBoost...")
plt.figure(figsize=(10, 6))
df_xgb_sorted = df_xgb.sort_values(by='standard', ascending=False)
plt.plot(df_xgb_sorted.index, df_xgb_sorted['standard'], marker='o', label='Standard', color='blue')
plt.plot(df_xgb_sorted.index, df_xgb_sorted['minmax'], marker='s', label='MinMax', color='red', linestyle='--')
plt.title('Importància XGBoost', fontsize=14)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'xgboost_importance_comparison.png'))
plt.close()

# ==============================================================================
# 5. RESUM DE RECOMANACIÓ (NOU)
# ==============================================================================
def summarize_best_variables(X_raw, column_names, df_xgb_results):
    print("\n--- 5. Sintetitzant les Millors Variables ---")
    
    # 1. Càlcul puntuació PCA (Max Loading Absolut en Standard)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_raw)
    pca = PCA(n_components=4).fit(X_sc)
    pca_scores = pd.Series(np.max(np.abs(pca.components_), axis=0), index=column_names)
    
    # 2. Càlcul puntuació XGBoost (fem servir standard, són iguals)
    xgb_scores = df_xgb_results['standard']
    
    # 3. Top 5 de cadascun
    top_pca = pca_scores.sort_values(ascending=False).head(5)
    top_xgb = xgb_scores.sort_values(ascending=False).head(5)
    
    # 4. Intersecció (Variables Clau)
    key_vars = list(set(top_pca.index) & set(top_xgb.index))
    
    # Guardar al fitxer
    with open(txt_output_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write("CONCLUSIÓ FINAL: RECOMANACIÓ DE VARIABLES\n")
        f.write(f"{'='*60}\n\n")
        
        f.write("TOP 5 VARIABLES SEGONS ESTRUCTURA DE DADES (PCA - Variància):\n")
        for v, s in top_pca.items(): f.write(f"  - {v:20} (Max Loading: {s:.4f})\n")
            
        f.write("\nTOP 5 VARIABLES SEGONS PREDICCIÓ (XGBoost - Importància):\n")
        for v, s in top_xgb.items(): f.write(f"  - {v:20} (Score: {s:.4f})\n")
            
        f.write("\n>>> VARIABLES 'GOLDEN' (Coincideixen en tots dos mètodes) <<<\n")
        f.write("Aquestes són les candidates més robustes per al model final:\n")
        if key_vars:
            for v in key_vars: f.write(f"  [X] {v}\n")
        else:
            f.write("  Cap coincidència directa al Top 5 (revisar Top 7).\n")
            
    print(f"Resum de recomanacions afegit a: {txt_output_path}")

# Executar el resum
summarize_best_variables(X, cols, df_xgb)

# ==============================================================================
# 6. AVALUACIÓ CLUSTERING
# ==============================================================================
print("\n--- 3. Avaluant Llindars i Silhouette Score ---")

def evaluate_thresholds(X_original, column_names):
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    scalers = ['minmax', 'standard']
    n_pca_comps_selection = 4
    k_clusters = 4
    
    results = []
    
    for scaler_name in scalers:
        scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
        X_scaled_global = scaler.fit_transform(X_original)
        
        pca = PCA(n_components=n_pca_comps_selection)
        pca.fit(X_scaled_global)
        loadings_abs = np.abs(pca.components_)
        max_loading_per_feature = np.max(loadings_abs, axis=0)
        
        for th in thresholds:
            selected_indices = np.where(max_loading_per_feature >= th)[0]
            
            if len(selected_indices) < 2:
                continue
            
            current_vars = [column_names[i] for i in selected_indices]
            X_subset = X_scaled_global[:, selected_indices]
            
            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_subset)
            
            sil_score = silhouette_score(X_subset, labels)
            
            # NOTA: Hem eliminat el print per consola de cada iteració com has demanat.
            
            results.append({
                'scaler': scaler_name,
                'threshold': th,
                'num_vars': len(current_vars),
                'vars': ", ".join(current_vars),
                'silhouette': sil_score
            })
            
    return pd.DataFrame(results)

df_eval = evaluate_thresholds(X, cols)
df_eval.to_csv(os.path.join(output_folder, 'evaluacio_variables_silhouette.csv'), index=False)
print("Càlculs finalitzats. CSV guardat.")

# Gràfic Silhouette
print("\n--- 4. Generant Gràfic Comparatiu Final ---")
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.lineplot(data=df_eval, x='threshold', y='silhouette', hue='scaler', 
             marker='o', linewidth=2.5, palette=['blue', 'red'])
plt.title('Qualitat dels Clusters vs Llindar', fontsize=14)
plt.ylabel('Silhouette Score', fontsize=12)
plt.xlabel('Threshold', fontsize=12)

for i in range(df_eval.shape[0]):
    row = df_eval.iloc[i]
    plt.text(row['threshold'], row['silhouette']+0.005, f"v={int(row['num_vars'])}", 
             ha='center', size='small', weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'grafic_comparativa_silhouette.png'))
plt.close()

print(f"Gràfic final guardat. Procés completat.")