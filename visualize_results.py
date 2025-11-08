# ============================================
# Visualization Script for Heart Disease Project
# ============================================

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# --- Create directory for saving figures ---
os.makedirs("visualizations", exist_ok=True)

# --- Load Results ---
with open("supervised_results.json", "r") as f:
    supervised = json.load(f)

with open("unsupervised_results.json", "r") as f:
    unsupervised = json.load(f)

# --- Convert Supervised Results to DataFrame ---
supervised_df = pd.DataFrame(supervised).T.reset_index()
supervised_df.rename(columns={"index": "Model"}, inplace=True)

# --- Convert Unsupervised Results to DataFrame ---
unsupervised_df = pd.DataFrame(unsupervised).T.reset_index()
unsupervised_df.rename(columns={"index": "Model"}, inplace=True)

# ==============================
# 1️⃣ Supervised Models - Accuracy Comparison
# ==============================
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="accuracy", data=supervised_df, palette="Blues_d")
plt.title("Supervised Models - Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("visualizations/supervised_accuracy.png")
plt.close()

# ==============================
# 2️⃣ Supervised Models - F1 Score Comparison
# ==============================
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="f1_score", data=supervised_df, palette="Greens_d")
plt.title("Supervised Models - F1 Score Comparison")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.savefig("visualizations/supervised_f1.png")
plt.close()

# ==============================
# 3️⃣ Unsupervised Models - Silhouette Score Comparison
# ==============================
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="silhouette_score", data=unsupervised_df, palette="Purples_d")
plt.title("Unsupervised Models - Silhouette Score Comparison")
plt.xlabel("Model")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig("visualizations/unsupervised_silhouette.png")
plt.close()

# ==============================
# 4️⃣ Unsupervised Models - Davies-Bouldin Score
# ==============================
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="davies_bouldin_score", data=unsupervised_df, palette="Reds_d")
plt.title("Unsupervised Models - Davies-Bouldin Score (Lower is Better)")
plt.xlabel("Model")
plt.ylabel("Davies-Bouldin Score")
plt.tight_layout()
plt.savefig("visualizations/unsupervised_davies_bouldin.png")
plt.close()

# ==============================
# 5️⃣ Combined Radar Chart (Supervised)
# ==============================

# Only use metrics that exist in the supervised results
available_cols = supervised_df.columns
categories = [c for c in ["accuracy", "precision", "recall", "f1_score", "roc_auc"] if c in available_cols]
N = len(categories)

plt.figure(figsize=(6, 6))
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

for i, row in supervised_df.iterrows():
    values = [row[c] for c in categories]
    values += values[:1]  # close the loop
    plt.polar(angles, values, label=row["Model"], linewidth=2)
plt.title("Supervised Models - Radar Chart", size=13)
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("visualizations/supervised_radar.png")
plt.close()

# ==============================
# 6️⃣ Summary Printout
# ==============================
print("✅ Visualization complete!")
print("Figures saved in 'visualizations/' folder:")
print(" - supervised_accuracy.png")
print(" - supervised_f1.png")
print(" - unsupervised_silhouette.png")
print(" - unsupervised_davies_bouldin.png")
print(" - supervised_radar.png")
