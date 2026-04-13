# =========================
# INSTALLS
# =========================
#!pip install pandas matplotlib seaborn numpy

# =========================
# IMPORTS
# =========================
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Rutgers colors
SCARLET = "#CC0033"
BLACK = "#000000"
GRAY = "#5F6A72"
LIGHT_GRAY = "#D0D3D4"

sns.set(style="whitegrid")

# =========================
# LOAD DATA
# =========================
lab = pd.read_json(BASE_DIR / "blood_data.json")
ref = pd.read_json(BASE_DIR / "blood_data_reference_ranges.json")
clinical = pd.read_json(BASE_DIR / "clinical_data.json")
path = pd.read_json(BASE_DIR / "pathological_data.json")

# =========================
# CLEAN / FORMAT
# =========================
lab["value"] = pd.to_numeric(lab["value"], errors="coerce")

# Pivot lab data (patients x analytes)
lab_pivot = lab.pivot_table(
    index="patient_id",
    columns="analyte_name",
    values="value",
    aggfunc="mean"
)

# Merge all datasets
df = lab_pivot.merge(clinical, on="patient_id", how="left")
df = df.merge(path, on="patient_id", how="left")

# =========================
# 1. AGE DISTRIBUTION
# =========================
plt.figure()
plt.hist(df["age_at_initial_diagnosis"].dropna(), bins=10)
plt.title("Age Distribution", color=BLACK)
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("age_distribution.png", dpi=300)

# =========================
# 2. SURVIVAL STATUS
# =========================
plt.figure()
df["survival_status"].value_counts().plot(kind="bar")
plt.title("Survival Status", color=BLACK)
plt.xticks(rotation=0)
plt.savefig("survival_status.png", dpi=300)

# =========================
# 3. SMOKING VS RECURRENCE
# =========================
ct = pd.crosstab(df["smoking_status"], df["recurrence"])

ct.plot(kind="bar", stacked=True)
plt.title("Smoking vs Recurrence", color=BLACK)
plt.savefig("smoking_vs_recurrence.png", dpi=300)

# =========================
# 4. LAB HEATMAP
# =========================
plt.figure(figsize=(12, 6))
sns.heatmap(lab_pivot.fillna(0), cmap="coolwarm")
plt.title("Lab Values Heatmap")
plt.savefig("lab_heatmap.png", dpi=300)

# =========================
# 5. ANALYTE DISTRIBUTION
# =========================
top_analytes = lab["analyte_name"].value_counts().index[:5]

for analyte in top_analytes:
    plt.figure()
    subset = lab[lab["analyte_name"] == analyte]
    plt.hist(subset["value"].dropna(), bins=20)
    plt.title(f"{analyte} Distribution")
    plt.savefig(f"{analyte}_dist.png", dpi=300)

# =========================
# 6. STAGE VS SURVIVAL
# =========================
plt.figure()
pd.crosstab(df["pT_stage"], df["survival_status"]).plot(kind="bar", stacked=True)
plt.title("Tumor Stage vs Survival")
plt.savefig("stage_vs_survival.png", dpi=300)

# =========================
# 7. HPV VS TUMOR SITE
# =========================
plt.figure()
pd.crosstab(df["primary_tumor_site"], df["hpv_association_p16"]).plot(kind="bar", stacked=True)
plt.title("Tumor Site vs HPV Status")
plt.savefig("hpv_vs_site.png", dpi=300)

# =========================
# 8. LYMPH NODES VS RECURRENCE
# =========================
plt.figure()
sns.scatterplot(
    x=df["number_of_positive_lymph_nodes"],
    y=df["days_to_recurrence"],
)
plt.title("Lymph Nodes vs Time to Recurrence")
plt.savefig("lymph_vs_recurrence.png", dpi=300)

# =========================
# SAVE CLEAN DATASET
# =========================
df.to_csv("merged_clinical_dataset.csv", index=False)