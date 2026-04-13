import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))

import pandas as pd
import numpy as np
import matplotlib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import umap
except Exception:
    raise ImportError("Install umap-learn with: pip install umap-learn")

try:
    import shap
except Exception:
    raise ImportError("Install shap with: pip install shap")

SCARLET = "#CC0033"
BLACK = "#000000"
GRAY = "#5F6A72"
LIGHT_GRAY = "#D0D3D4"

lab = pd.read_json(PROJECT_DIR / "blood_data.json")
ref = pd.read_json(PROJECT_DIR / "blood_data_reference_ranges.json")
clinical = pd.read_json(PROJECT_DIR / "clinical_data.json")
path = pd.read_json(PROJECT_DIR / "pathological_data.json")

lab["patient_id"] = lab["patient_id"].astype(str)
clinical["patient_id"] = clinical["patient_id"].astype(str)
path["patient_id"] = path["patient_id"].astype(str)
lab["value"] = pd.to_numeric(lab["value"], errors="coerce")

lab_pivot = lab.pivot_table(index="patient_id", columns="analyte_name", values="value", aggfunc="mean").reset_index()
df = clinical.merge(path, on="patient_id", how="left").merge(lab_pivot, on="patient_id", how="left")
df["recurrence_binary"] = df["recurrence"].map({"yes": 1, "no": 0})

exclude_cols = ['patient_id', 'recurrence', 'recurrence_binary', 'survival_status', 'survival_status_with_cause']
numeric_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in df.columns if c not in exclude_cols and not pd.api.types.is_numeric_dtype(df[c])]

numeric_df = df[numeric_cols].copy()
categorical_df = pd.get_dummies(df[categorical_cols], dummy_na=True, drop_first=False)
X_full = pd.concat([numeric_df, categorical_df], axis=1)
X_full = X_full.loc[:, X_full.notna().sum() > 0]
y = df["recurrence_binary"]

mask = y.notna()
X = X_full.loc[mask].copy()
y = y.loc[mask].astype(int)

imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X.columns, index=X.index)

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "recurrence": y.map({0: "No recurrence", 1: "Recurrence"}).values
}, index=X.index)

plt.figure(figsize=(7, 5))
for label, color in [("No recurrence", GRAY), ("Recurrence", SCARLET)]:
    sub = pca_df[pca_df["recurrence"] == label]
    plt.scatter(sub["PC1"], sub["PC2"], label=label, alpha=0.85, s=50, color=color)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("PCA of merged clinical-pathology-lab features")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(PROJECT_DIR / "rutgers_pca.png", dpi=300)

# UMAP
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=min(10, max(2, len(X_scaled) - 1)),
    min_dist=0.2,
    random_state=42
)
X_umap = reducer.fit_transform(X_scaled)
umap_df = pd.DataFrame({
    "UMAP1": X_umap[:, 0],
    "UMAP2": X_umap[:, 1],
    "recurrence": y.map({0: "No recurrence", 1: "Recurrence"}).values
}, index=X.index)

plt.figure(figsize=(7, 5))
for label, color in [("No recurrence", GRAY), ("Recurrence", SCARLET)]:
    sub = umap_df[umap_df["recurrence"] == label]
    plt.scatter(sub["UMAP1"], sub["UMAP2"], label=label, alpha=0.85, s=50, color=color)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP of merged clinical-pathology-lab features")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(PROJECT_DIR / "rutgers_umap.png", dpi=300)

# Random forest + SHAP
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=2,
    random_state=42
)
rf.fit(X_imp, y)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_imp)

if isinstance(shap_values, list):
    sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
else:
    sv = shap_values
    if sv.ndim == 3:
        sv = sv[:, :, 1]

mean_abs_shap = np.abs(sv).mean(axis=0)
shap_imp = pd.Series(mean_abs_shap, index=X_imp.columns).sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 6))
plt.barh(shap_imp.index[::-1], shap_imp.values[::-1], color=GRAY)
plt.xlabel("Mean |SHAP value|")
plt.title("Top SHAP features for recurrence")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "rutgers_shap_importance.png", dpi=300)

print("\nTop SHAP features:")
print(shap_imp.head(10))
