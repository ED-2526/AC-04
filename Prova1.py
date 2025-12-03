import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import warnings
import matplotlib

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# Data loading
print("Loading dataset...")
try:
    data = pd.read_csv('marketing_campaign.csv', sep="\t")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: File 'marketing_campaign.csv' not found.")
    exit()

# Feature engineering
print("\nPreprocessing data...")
data['Age'] = 2025 - data['Year_Birth']
data['Total_Spending'] = (
    data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] +
    data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']
)
data['Has_Partner'] = data['Marital_Status'].isin(['Married', 'Together']).astype(int)
data['Family_Size'] = 1 + data['Has_Partner'] + data['Kidhome'] + data['Teenhome']

# Data cleaning
data = data.dropna(subset=['Income'])
data = data[data['Age'] < 100]
data = data[data['Income'] < 600000]
data = data[~data['Marital_Status'].isin(['YOLO', 'Absurd', 'Alone'])]

# Prepare features
df_cat = pd.get_dummies(data[['Education', 'Marital_Status']], drop_first=True)
df_features = pd.concat([data[["Age", "Total_Spending", "Income", "Family_Size"]], df_cat], axis=1)
X = df_features.values
cols = list(df_features.columns)

print(f"Data prepared. Shape: {X.shape}")

# PCA analysis function
def run_pca_analysis(X_input, scaler_name, n_components, column_names):
    print(f"\n{'='*60}")
    print(f"PCA ANALYSIS: {scaler_name.upper()}")
    print(f"{'='*60}")

    # Scale data
    scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
    X_scaled = scaler.fit_transform(X_input)

    # Standard deviation analysis
    print(f"\nStandard Deviation ({scaler_name}):")
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
    
    img_file = f"scree_plot_{scaler_name}.png"
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
    
    print(f"\nComponent Loadings ({scaler_name.upper()}):")
    sorted_data = {}

    for i in range(n_comps):
        pc_name = f"PC{i+1}"
        sorted_series = loadings_df[pc_name].abs().sort_values(ascending=False)
        
        print(f"\n{pc_name}:")
        print(sorted_series.head(6))
        
        original_values = loadings_df.loc[sorted_series.index, pc_name]
        sorted_data[f"{pc_name}_Variable"] = sorted_series.index.tolist()
        sorted_data[f"{pc_name}_Value"] = original_values.tolist()

    # Save results
    csv_file = f"loadings_{scaler_name}.csv"
    pd.DataFrame(sorted_data).to_csv(csv_file, index=False)
    print(f"\nResults saved: {csv_file}")

    # Cumulative variance
    cumulative_var = pca.explained_variance_ratio_.cumsum()[-1] * 100
    print(f"\nCumulative variance with {n_comps} components: {cumulative_var:.2f}%")

# Run analyses
run_pca_analysis(X, 'minmax', n_components=7, column_names=cols)
run_pca_analysis(X, 'standard', n_components=9, column_names=cols)

print(f"\n{'='*60}")
print("Analysis completed!")
print(f"{'='*60}")