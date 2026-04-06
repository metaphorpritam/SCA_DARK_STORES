import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# MOCK FEATURE IMPORTANCE
# (Replace later if model available)
# -----------------------------
features = [
    "Delivery Delay",
    "Freight Value",
    "Product Weight",
    "Review Score",
    "Product Category",
]

importance = [0.35, 0.25, 0.15, 0.15, 0.10]

df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values(
    by="Importance", ascending=False
)

# -----------------------------
# PLOT
# -----------------------------
plt.figure()
plt.bar(df["Feature"], df["Importance"])
plt.xticks(rotation=30)
plt.title("Feature Importance for Returns")
plt.ylabel("Importance Score")

plt.tight_layout()

# -----------------------------
# SAVE
# -----------------------------
plt.savefig("outputs/feature_importance.png")

print("✅ feature_importance.png created")
