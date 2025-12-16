import os
import shutil
import warnings
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

from xgboost import XGBRegressor

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

from yellowbrick.cluster import KElbowVisualizer
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from matplotlib.colors import ListedColormap, BoundaryNorm

# ==============================================================================
# 0. CONFIGURACIÓ INICIAL
# ==============================================================================
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

OUTPUT_DIR = "resultats_xgboost_silhouette"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Carpeta '{OUTPUT_DIR}' preparada.")

MIN_VARS = 4
K_MIN, K_MAX = 2, 12
ELBOW_METRIC = "distortion"

# ==============================================================================
# 1. CÀRREGA I PREPROCESSAMENT
# ==============================================================================
print("\n--- 1. Càrrega i Preprocessament de Dades ---")
filename = "marketing_campaign.csv"
data = pd.read_csv(filename, sep="\t")

data["Age"] = 2025 - data["Year_Birth"]
data["Total_Spending"] = (
    data["MntWines"] + data["MntFruits"] +
    data["MntMeatProducts"] + data["MntFishProducts"] +
    data["MntSweetProducts"] + data["MntGoldProds"]
)

# Outliers Total_Spending
Q1_spend = data["Total_Spending"].quantile(0.25)
Q3_spend = data["Total_Spending"].quantile(0.75)
IQR_spend = Q3_spend - Q1_spend
data = data[data["Total_Spending"] <= (Q3_spend + 2.0 * IQR_spend)]

# Family
partner_status = ["Married", "Together"]
data["Has_Partner"] = data["Marital_Status"].apply(lambda x: 1 if x in partner_status else 0)
data["Family_Size"] = 1 + data["Has_Partner"] + data["Kidhome"] + data["Teenhome"]

# Tenure
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], dayfirst=True)
data["Tenure_Days"] = (data["Dt_Customer"].max() - data["Dt_Customer"]).dt.days

# Neteja bàsica
data = data.dropna(subset=["Income"])
data = data[(data["Age"] < 100) & (data["Income"] < 600000)]
invalid_status = ["YOLO", "Absurd", "Alone"]
data = data[~data["Marital_Status"].isin(invalid_status)]

# Outliers Income
Q1_inc = data["Income"].quantile(0.25)
Q3_inc = data["Income"].quantile(0.75)
data = data[data["Income"] <= (Q3_inc + 1.5 * (Q3_inc - Q1_inc))]

# Ratios
eps = 1e-6
data["Wine_Ratio"] = data["MntWines"] / (data["Total_Spending"] + eps)
data["Meat_Ratio"] = data["MntMeatProducts"] / (data["Total_Spending"] + eps)
data["Sweet_Ratio"] = data["MntSweetProducts"] / (data["Total_Spending"] + eps)
data["Fish_Ratio"] = data["MntFishProducts"] / (data["Total_Spending"] + eps)
data["Fruit_Ratio"] = data["MntFruits"] / (data["Total_Spending"] + eps)
data["Gold_Ratio"] = data["MntGoldProds"] / (data["Total_Spending"] + eps)

# Encoding categòric (per XGBoost)
data["Education_Code"], edu_uniques = pd.factorize(data["Education"], sort=True)
data["Marital_Status_Code"], mar_uniques = pd.factorize(data["Marital_Status"], sort=True)

print(f"Dades netes: {len(data)} registres.")
print("Mapping Education:", {cat: i for i, cat in enumerate(edu_uniques)})
print("Mapping Marital_Status:", {cat: i for i, cat in enumerate(mar_uniques)})

# ==============================================================================
# 2. VARIABLES
# ==============================================================================
print("\n--- 2. Definició de variables ---")
target_col = "Total_Spending"

xgb_feature_cols = [
    "Income",
    "MntWines", "MntMeatProducts", "MntFishProducts",
    "MntFruits", "MntSweetProducts", "MntGoldProds",
    "Wine_Ratio", "Meat_Ratio", "Sweet_Ratio", "Fish_Ratio", "Fruit_Ratio", "Gold_Ratio",
    "Tenure_Days", "Family_Size", "Age",
    "Education_Code", "Marital_Status_Code"
]

cluster_feature_cols = [
    "Income",
    "MntWines", "MntMeatProducts", "MntFishProducts",
    "MntFruits", "MntSweetProducts", "MntGoldProds",
    "Wine_Ratio", "Meat_Ratio", "Sweet_Ratio", "Fish_Ratio", "Fruit_Ratio", "Gold_Ratio",
    "Tenure_Days", "Family_Size", "Age"
]

print(f"Features XGBoost ({len(xgb_feature_cols)}): {xgb_feature_cols}")
print(f"Features Clustering ({len(cluster_feature_cols)}): {cluster_feature_cols}")

# ==============================================================================
# 3. XGBOOST (4 casos) + CONSENS
# ==============================================================================
print("\n--- 3. XGBoost supervisat (4 casos) ---")

def compute_xgb_importance(data_df, feature_cols, target_col, scaler_name, importance_method, random_state=42):
    X = data_df[feature_cols].values
    y = data_df[target_col].values

    scaler = StandardScaler() if scaler_name == "standard" else MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=random_state
    )

    model = XGBRegressor(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        objective="reg:squarederror",
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    if importance_method == "gain":
        importances = model.feature_importances_
    elif importance_method == "permutation":
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=10,
            random_state=random_state, n_jobs=-1
        )
        importances = perm.importances_mean
    else:
        raise ValueError("importance_method ha de ser 'gain' o 'permutation'")

    df_imp = pd.DataFrame({"feature": feature_cols, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    return df_imp, (r2, rmse)

cases = [("standard","gain"), ("standard","permutation"), ("minmax","gain"), ("minmax","permutation")]
imp_results, imp_metrics = {}, {}

for sc, m in cases:
    df_imp, (r2, rmse) = compute_xgb_importance(data, xgb_feature_cols, target_col, sc, m)
    imp_results[(sc, m)] = df_imp
    imp_metrics[(sc, m)] = (r2, rmse)
    df_imp.to_csv(os.path.join(OUTPUT_DIR, f"xgb_importance_{m}_{sc}.csv"), index=False)
    print(f"[{sc.upper()} + {m.upper()}] R²={r2:.4f} | RMSE={rmse:.2f}")

# Plot 2x2 importàncies
titles = {
    ("standard","gain"): "STANDARD + GAIN",
    ("standard","permutation"): "STANDARD + PERMUTATION",
    ("minmax","gain"): "MINMAX + GAIN",
    ("minmax","permutation"): "MINMAX + PERMUTATION"
}
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.ravel()
top_n_plot = len(xgb_feature_cols)

for i, (sc, m) in enumerate(cases):
    ax = axes[i]
    df_case = imp_results[(sc, m)].head(top_n_plot).iloc[::-1]
    ax.barh(df_case["feature"], df_case["importance"])
    ax.set_title(titles[(sc, m)], fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("Importance" if i in [2,3] else "")
    ax.set_ylabel("Feature" if i in [0,2] else "")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"comparativa_importancia_4casos_top{top_n_plot}.png"), dpi=170)
plt.close()

# Consens ponderat per R² (només features de clustering)
cons_features = [f for f in cluster_feature_cols if f in xgb_feature_cols]
ranking_df = pd.DataFrame({"feature": cons_features})

for sc, m in cases:
    df_case = imp_results[(sc, m)].copy()
    df_case = df_case[df_case["feature"].isin(cons_features)].reset_index(drop=True)
    df_case["rank"] = np.arange(1, len(df_case) + 1)
    rank_map = dict(zip(df_case["feature"], df_case["rank"]))
    ranking_df[f"rank_{sc}_{m}"] = ranking_df["feature"].map(rank_map)

rank_cols = [c for c in ranking_df.columns if c.startswith("rank_")]
ranking_df["rank_mean"] = ranking_df[rank_cols].mean(axis=1)

weights = np.array([imp_metrics[(sc, m)][0] for sc, m in cases], dtype=float)
weights = np.maximum(weights, 1e-6)
weights = weights / weights.sum()

rank_matrix = ranking_df[rank_cols].values
ranking_df["rank_weighted_by_r2"] = (rank_matrix * weights).sum(axis=1)
ranking_df = ranking_df.sort_values("rank_weighted_by_r2").reset_index(drop=True)

ranking_df.to_csv(os.path.join(OUTPUT_DIR, "ranking_consens_xgb_4cases.csv"), index=False)
ranked_features = ranking_df["feature"].tolist()

plt.figure(figsize=(10, 6))
top_show = min(15, len(ranked_features))
show_df = ranking_df.head(top_show).copy()
plt.barh(show_df["feature"][::-1], (-show_df["rank_weighted_by_r2"])[::-1])
plt.title("Top variables (consens XGBoost 4 casos) [més a dalt = més important]")
plt.xlabel("Score (− rank ponderat per R²)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ranking_consens_top.png"), dpi=170)
plt.close()

print("\nTop 10 variables consens (clustering):", ranked_features[:10])

# ==============================================================================
# 4. FUNCIONS AUXILIARS
# ==============================================================================
def scale_data(X_df, scaler_name):
    scaler = StandardScaler() if scaler_name == "standard" else MinMaxScaler()
    return scaler.fit_transform(X_df.values)

def k_optima_and_save_elbow(X_scaled, k_min=2, k_max=12, metric="distortion", outpath=None, title="Elbow"):
    model = KMeans(random_state=42, n_init=10)
    viz = KElbowVisualizer(model, k=(k_min, k_max), metric=metric, timings=False)
    viz.fit(X_scaled)
    best_k = viz.elbow_value_
    if best_k is None:
        try:
            best_k = int(viz.k_values_[int(np.argmin(viz.metric_values_))])
        except Exception:
            best_k = 4
    if outpath is not None:
        viz.show(outpath=outpath)
        plt.close()
    return int(best_k)

def fit_predict(method, X_scaled, k):
    if method == "kmeans":
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        return model.fit_predict(X_scaled)

    if method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        return model.fit_predict(X_scaled)

    if method == "gmm":
        model = GaussianMixture(n_components=k, random_state=42, covariance_type="full")
        return model.fit_predict(X_scaled)

    if method == "spectral":
        n_neighbors = min(15, max(5, int(np.sqrt(len(X_scaled)))))
        model = SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            random_state=42,
            assign_labels="kmeans"
        )
        return model.fit_predict(X_scaled)

    raise ValueError("method ha de ser kmeans/hierarchical/gmm/spectral")

def compute_centroids(X_scaled, labels, uniq_labels):
    centroids = np.zeros((len(uniq_labels), X_scaled.shape[1]))
    for i, c in enumerate(uniq_labels):
        pts = X_scaled[labels == c]
        centroids[i] = pts.mean(axis=0) if len(pts) else np.zeros(X_scaled.shape[1])
    return centroids

def sse_bss_normalized(X_scaled, labels):
    uniq = np.unique(labels)
    overall = X_scaled.mean(axis=0, keepdims=True)
    centroids = compute_centroids(X_scaled, labels, uniq)

    # SSE
    sse = 0.0
    for i, c in enumerate(uniq):
        pts = X_scaled[labels == c]
        if len(pts) == 0:
            continue
        sse += ((pts - centroids[i]) ** 2).sum()

    # BSS
    bss = 0.0
    for i, c in enumerate(uniq):
        n_c = np.sum(labels == c)
        if n_c == 0:
            continue
        bss += n_c * ((centroids[i] - overall.squeeze()) ** 2).sum()

    tss = sse + bss if (sse + bss) > 0 else 1e-12
    return sse/tss, bss/tss

def corr_distance_to_centroid_vs_spending(X_scaled, labels, spending):
    uniq = np.unique(labels)
    centroids = compute_centroids(X_scaled, labels, uniq)

    d = np.zeros(len(X_scaled))
    for idx in range(len(X_scaled)):
        c = labels[idx]
        pos = np.where(uniq == c)[0][0]
        d[idx] = np.linalg.norm(X_scaled[idx] - centroids[pos])

    if np.std(d) < 1e-12 or np.std(spending) < 1e-12:
        return np.nan
    return pearsonr(d, spending)[0]

def cluster_profile_all_numeric(data_df, labels, cluster_colname):
    """
    Perfil complet com la imatge (mateixos labels que el t-SNE):
      - mitjanes de totes les columnes numèriques
      - + Count (Clients)
      - transposat (files variables, columnes clústers)
    """
    tmp = data_df.copy()
    tmp[cluster_colname] = labels

    num_cols = tmp.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != cluster_colname]

    means = tmp.groupby(cluster_colname)[num_cols].mean().round(2)
    counts = tmp[cluster_colname].value_counts().sort_index()

    means.loc[:, "Count (Clients)"] = counts.values
    return means.T

def plot_tsne_pretty(X_scaled, labels, outpath, title):
    n = len(X_scaled)
    perplexity = min(35, max(8, n//60))
    perplexity = min(perplexity, n-1)

    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    uniq = np.unique(labels)
    k_eff = len(uniq)

    # colormap discret (k_eff colors)
    base = plt.cm.get_cmap("viridis", k_eff)
    cmap = ListedColormap([base(i) for i in range(k_eff)])
    # mapeig label->0..k_eff-1 per colorbar neta
    label_to_idx = {lab: i for i, lab in enumerate(uniq)}
    labels_idx = np.array([label_to_idx[x] for x in labels])

    norm = BoundaryNorm(np.arange(-0.5, k_eff + 0.5, 1), cmap.N)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    sc = ax.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=labels_idx, cmap=cmap, norm=norm,
        s=26, alpha=0.9, linewidths=0.25, edgecolors="k"
    )

    # centroides (aprox) per cada clúster real (uniq)
    centroids = compute_centroids(X_scaled, labels, uniq)
    idx_cent = np.argmin(cdist(centroids, X_scaled), axis=1)
    ax.scatter(
        X_tsne[idx_cent, 0], X_tsne[idx_cent, 1],
        s=160, marker="X",
        c=np.arange(k_eff), cmap=cmap, norm=norm,
        edgecolors="k", linewidths=1.0, alpha=0.95
    )

    # colorbar amb ticks exactes (0..k_eff-1) però que representen els clústers reals uniq
    cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(k_eff))
    cbar.set_label("Cluster")
    cbar.set_ticklabels([str(u) for u in uniq])

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

# ==============================================================================
# 5. GRID silhouette vs #vars
# ==============================================================================
print("\n--- 5. GRID silhouette vs #vars (Standard vs MinMax) ---")

X_base = pd.DataFrame(data[cluster_feature_cols].values, columns=cluster_feature_cols)
y_spend = data[target_col].values

methods = ["kmeans", "hierarchical", "gmm", "spectral"]
scalers = ["standard", "minmax"]

max_vars = len(ranked_features)
vars_list = list(range(MIN_VARS, max_vars + 1))

best_per_method = {}
all_rows = []

ELBOW_DIR = os.path.join(OUTPUT_DIR, "kelbow")
os.makedirs(ELBOW_DIR, exist_ok=True)

for method in methods:
    sil_by_scaler = {s: [] for s in scalers}
    best_global = {"silhouette": -1, "n_vars": None, "k": None, "features": None, "scaler": None}

    print(f"\n>> {method.upper()}")

    for scaler_name in scalers:
        for n_vars in vars_list:
            feats = ranked_features[:n_vars]
            X_sub = X_base[feats]
            X_scaled = scale_data(X_sub, scaler_name)

            elbow_path = os.path.join(ELBOW_DIR, f"elbow_{method}_{scaler_name}_vars{n_vars}.png")
            try:
                k = k_optima_and_save_elbow(
                    X_scaled,
                    k_min=K_MIN,
                    k_max=min(K_MAX, len(X_scaled) - 1),
                    metric=ELBOW_METRIC,
                    outpath=elbow_path,
                    title=f"Elbow | {method} | {scaler_name} | vars={n_vars}"
                )
            except Exception:
                k = 4

            try:
                labels = fit_predict(method, X_scaled, k)
                if len(np.unique(labels)) < 2:
                    sil = -1
                else:
                    sil = float(silhouette_score(X_scaled, labels))
            except Exception:
                sil = -1

            sil_by_scaler[scaler_name].append(sil)
            all_rows.append({"method": method, "scaler": scaler_name, "n_vars": n_vars, "k": k, "silhouette": sil})

            if sil > best_global["silhouette"]:
                best_global = {"silhouette": sil, "n_vars": n_vars, "k": k, "features": feats, "scaler": scaler_name}

    best_per_method[method] = best_global
    print(f"  -> Millor GLOBAL: sil={best_global['silhouette']:.4f} | scaler={best_global['scaler']} | vars={best_global['n_vars']} | k={best_global['k']}")

    plt.figure()
    plt.plot(vars_list, sil_by_scaler["standard"], marker="o", label="StandardScaler")
    plt.plot(vars_list, sil_by_scaler["minmax"], marker="o", label="MinMaxScaler")
    plt.title(f"Silhouette vs #Variables | {method}")
    plt.xlabel("Nombre de variables (top-n importància XGB consens)")
    plt.ylabel("Silhouette")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"silhouette_{method}_standard_vs_minmax.png"), dpi=170)
    plt.close()

pd.DataFrame(all_rows).to_csv(os.path.join(OUTPUT_DIR, "grid_silhouette_all.csv"), index=False)

# ==============================================================================
# 6. MILLORS MODELS: mètriques + t-SNE + PERFIL COMPLET (mateixos labels)
# ==============================================================================
print("\n--- 6. Millors models (1 per mètode) + mètriques + t-SNE + perfil ---")

FINAL_DIR = os.path.join(OUTPUT_DIR, "best_models")
os.makedirs(FINAL_DIR, exist_ok=True)

PROFILE_DIR = os.path.join(FINAL_DIR, "profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)

for method in methods:
    best = best_per_method[method]
    feats = best["features"]
    scaler_name = best["scaler"]
    k = best["k"]
    n_vars = best["n_vars"]

    X_sub = X_base[feats]
    X_scaled = scale_data(X_sub, scaler_name)
    labels = fit_predict(method, X_scaled, k)

    uniq = np.unique(labels)
    k_eff = len(uniq)  # <-- IMPORTANT: clústers reals dels labels (t-SNE i perfil igual)

    # Si no hi ha com a mínim 2 clústers reals, saltem
    if k_eff < 2:
        print(f"[SKIP] {method}: només {k_eff} clúster real.")
        continue

    sil = float(silhouette_score(X_scaled, labels))
    sse_n, bss_n = sse_bss_normalized(X_scaled, labels)
    corr = corr_distance_to_centroid_vs_spending(X_scaled, labels, y_spend)

    print(f"\n[{method.upper()}] scaler={scaler_name} | vars={n_vars} | k(elbow)={k} | k(efectiu)={k_eff}")
    print(f"  features: {feats}")
    print(f"  silhouette: {sil:.4f}")
    print(f"  SSE_norm:   {sse_n:.4f}")
    print(f"  BSS_norm:   {bss_n:.4f}")
    print(f"  corr(dist->centroid, spending): {corr:.4f}")

    # counts per clúster (mateixos que “veus”)
    counts = pd.Series(labels).value_counts().sort_index()
    print("  Count per clúster:")
    print(counts.to_string())

    # t-SNE (mateixos labels)
    tsne_path = os.path.join(FINAL_DIR, f"tsne_{method}_{scaler_name}_k{k}_keff{k_eff}_vars{n_vars}.png")
    title = f"t-SNE - {method.upper()}\nscaler={scaler_name} | k(elbow)={k} | k(efectiu)={k_eff} | vars={n_vars} | sil={sil:.3f}"
    plot_tsne_pretty(X_scaled, labels, tsne_path, title)

    # PERFIL COMPLET (mateixos labels)
    cluster_col = f"{method.upper()}_Cluster"
    profile_T = cluster_profile_all_numeric(data, labels, cluster_colname=cluster_col)

    header = (
        "=" * 110 + "\n"
        f"PERFIL COMPLET: {method.upper()} | scaler={scaler_name} | k(elbow)={k} | k(efectiu)={k_eff} | vars={n_vars}\n"
        f"silhouette={sil:.4f}\n"
        f"Features clustering: {', '.join(feats)}\n"
        + "=" * 110
    )

    print("\n" + header)
    print(profile_T)

    out_txt = os.path.join(PROFILE_DIR, f"perfil_{method}_{scaler_name}_k{k}_keff{k_eff}_vars{n_vars}.txt")
    out_csv = os.path.join(PROFILE_DIR, f"perfil_{method}_{scaler_name}_k{k}_keff{k_eff}_vars{n_vars}.csv")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(header + "\n\n")
        f.write(profile_T.to_string())
        f.write("\n")
    profile_T.to_csv(out_csv, sep="\t")

print("\n✅ PROCÉS COMPLETAT!")
print(f"📁 Resultats a: {OUTPUT_DIR}")
