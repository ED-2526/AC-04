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

# Age = any actual - Year_Birth
data['Age'] = 2025 - data['Year_Birth']

# Total Spending = suma de totes les despeses
data['Total_Spending'] = (
    data['MntWines'] +
    data['MntFruits'] +
    data['MntMeatProducts'] +
    data['MntFishProducts'] +
    data['MntSweetProducts'] +
    data['MntGoldProds']
)

# Family size = Kidhome + Teenhome
partner_status = ['Married', 'Together']
data['Has_Partner'] = data['Marital_Status'].apply(lambda x: 1 if x in partner_status else 0)
data['Family_Size'] = 1 + data['Has_Partner'] + data['Kidhome'] + data['Teenhome']

# --- NETEJA ---
data = data.dropna(subset=['Income'])
data = data[data['Age'] < 100]
data = data[data['Income'] < 600000]
invalid_status = ['YOLO', 'Absurd', 'Alone']
data = data[~data['Marital_Status'].isin(invalid_status)]

# --- CODIFICACIÓ CATEGÒRICA ---
education_cols = ['Education']
marital_cols = ['Marital_Status']

df_cat = pd.get_dummies(
    data[education_cols + marital_cols],
    drop_first=True
)

# --- DEFINICIÓ DE LES VARIABLES FINALS ---
cols = [
    "Age",
    "Total_Spending",
    "Income",
    "Family_Size"
] + list(df_cat.columns)

print("\nColumnes finals utilitzades per clustering i anàlisi:")
print(cols)

# --- AJUNTAR NUMÈRIQUES + CATEGÒRIQUES CODIFICADES ---
df_features = pd.concat([
    data[["Age", "Total_Spending", "Income", "Family_Size"]],
    df_cat
], axis=1)

# --- ESCALAT ---
from sklearn.preprocessing import MinMaxScaler

X = df_features.values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("\nEscalat MinMax completat!")
print("\n========== CORRELACIÓ: Total_Spending vs Income ==========")

# Calcular correlació
correlation = data["Total_Spending"].corr(data["Income"])

print(f"Correlació Total_Spending - Income: {correlation:.4f}")

# -----------------------------------------------------
# 3. ANALISI DE VARIABLES: Desviació, Regressió i PCA
# -----------------------------------------------------

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

print("\n========== ANALISI DE VARIABLES ==========")

# ================================================
# A) DESVIACIÓ ESTÀNDARD
# ================================================
print("\n--- DESVIACIÓ ESTÀNDARD ---")
std_vals = df_features.std().sort_values(ascending=False)
print(std_vals)


# ================================================
# B) PCA: Loadings i variança
# ================================================
print("\n--- PCA (LOADINGS) ---")

n_components = min(3, len(cols))  
pca = PCA(n_components=n_components)
pca.fit(X_scaled)

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(n_components)],
    index=cols
)

print(loadings)

print("\nVariança explicada:", pca.explained_variance_ratio_)
print("Variança acumulada:", pca.explained_variance_ratio_.cumsum())
