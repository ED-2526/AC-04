import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import warnings
import matplotlib
import os

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# --- 1. DATA LOADING (DEL CSV PROCESSAT) ---
print("Loading processed dataset...")

# Ajusta el nom de la carpeta si és diferent, però al codi anterior era 'resultats_clustering'
input_file = 'dades_processades_completes.csv'

try:
    # Aquest fitxer ja el guardem separat per comes al codi principal
    data = pd.read_csv(input_file, sep=",")
    print(f"Dataset '{input_file}' loaded successfully!")
    print(f"Shape: {data.shape}")
except FileNotFoundError:
    print(f"Error: File '{input_file}' not found. Assegura't d'haver executat el pas anterior.")
    exit()

# --- 2. PREPARACIÓ DE FEATURES ---
# Com que el CSV ja té les dades netes i les variables creades (Age, Total_Spending, etc.),
# només hem de seleccionar les columnes que volem analitzar.

selected_columns = [
    "Age", 
    "Total_Spending", 
    "Income", 
    "Family_Size", 
    "Seniority_Code", 
    "Education_Code", 
    "Marital_Status_Code"
]

# Verifiquem que totes les columnes existeixin al CSV
missing_cols = [col for col in selected_columns if col not in data.columns]
if missing_cols:
    print(f"Error: Falten aquestes columnes al CSV carregat: {missing_cols}")
    exit()

# Creem el DataFrame de features
df_features = data[selected_columns]

X = df_features.values
cols = list(df_features.columns)

print(f"Data prepared for PCA. Shape: {X.shape}")
print(f"Columns used: {cols}")

# --- 3. PCA ANALYSIS FUNCTION (Igual que abans) ---
def run_pca_analysis(X_input, scaler_name, n_components, column_names):
    print(f"\n{'#'*60}")
    print(f"ANALYSIS WITH {scaler_name.upper()}")
    print(f"{'#'*60}")

    # Scale data
    scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
    X_scaled = scaler.fit_transform(X_input)

    # Standard deviation analysis
    print(f"\n{'_'*20}")
    print(f"\nStandard Deviation ({scaler_name}):")
    print(f"{'_'*20}")

    std_series = pd.Series(X_scaled.std(axis=0), index=column_names).sort_values(ascending=False)
    print(std_series.head(10))

    # Scree plot
    n_features = X_scaled.shape[1]
    n_plot = min(15, n_features)
    
    pca_plot = PCA(n_components=n_plot)
    pca_plot.fit(X_scaled)
    
    variance = pca_plot.explained_variance_ratio_ * 100
    components = range(1, len(variance) + 1)

    plt.figure(figsize=(10, 6))
    plt.bar(components, variance, alpha=0.9, color='#337ab7')
    plt.plot(components, variance, marker='o', linestyle='-', color='black', linewidth=1.5)
    plt.title(f'Scree Plot - {scaler_name.upper()}', pad=20)
    plt.xlabel('Principal Components')
    plt.ylabel('% Variance Explained')
    plt.xticks(components)
    plt.grid(axis='y', alpha=0.5)
    
    img_file = f"resultats_clustering/scree_plot_{scaler_name}_final.png"
    plt.savefig(img_file)
    plt.close()
    print(f"\nScree plot saved: {img_file}")

    # PCA with specified components
    n_comps = min(n_components, n_features)
    pca = PCA(n_components=n_comps)
    pca.fit(X_scaled)

    # Component loadings
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_comps)],
        index=column_names
    )
    print(f"\n{'_'*20}")
    print(f"\nPCA Component Loadings ({scaler_name}):")
    print(f"{'_'*20}")
    
    sorted_data = {}

    for i in range(n_comps):
        pc_name = f"PC{i+1}"
        sorted_series = loadings_df[pc_name].abs().sort_values(ascending=False)
        
        print(f"\n{pc_name}:")
        print(f"{'_'*10}")
        print(sorted_series.head(6))
        
        original_values = loadings_df.loc[sorted_series.index, pc_name]
        sorted_data[f"{pc_name}_Variable"] = sorted_series.index.tolist()
        sorted_data[f"{pc_name}_Value"] = original_values.tolist()


    # Cumulative variance
    cumulative_var = pca.explained_variance_ratio_.cumsum()[-1] * 100
    print(f"\nCumulative variance with {n_comps} components: {cumulative_var:.2f}%")

# --- 4. RUN PCA ANALYSIS WITH BOTH SCALERS ---
run_pca_analysis(X, 'minmax', n_components=4, column_names=cols)
run_pca_analysis(X, 'standard', n_components=6, column_names=cols)
