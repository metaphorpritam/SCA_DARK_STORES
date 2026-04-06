import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 10))
ax.axis("off")

# -----------------------------
# DEFINE BLOCKS WITH GROUPS
# -----------------------------
blocks = [
    ("Raw Data (Olist)", 7, "#4CAF50"),
    ("Data Pipeline", 6, "#4CAF50"),
    ("Feature Engineering", 5, "#2196F3"),
    ("Clustering (K-Means)", 4, "#2196F3"),
    ("Dark Store Selection", 3, "#FF9800"),
    ("Demand Visualization", 2, "#FF9800"),
    ("VRP Optimization", 1, "#9C27B0"),
    ("Final Dashboard", 0, "#9C27B0"),
]

# -----------------------------
# DRAW BOXES
# -----------------------------
for text, y, color in blocks:
    ax.text(
        0.5,
        y,
        text,
        ha="center",
        va="center",
        fontsize=12,
        color="white",
        bbox=dict(boxstyle="round,pad=0.5", fc=color, ec="black"),
    )

# -----------------------------
# DRAW ARROWS
# -----------------------------
for i in range(len(blocks) - 1):
    y1 = blocks[i][1]
    y2 = blocks[i + 1][1]

    ax.annotate(
        "",
        xy=(0.5, y2 + 0.3),
        xytext=(0.5, y1 - 0.3),
        arrowprops=dict(arrowstyle="->", lw=2, color="black"),
    )

# -----------------------------
# ADD SECTION LABELS (KEY UPGRADE 🔥)
# -----------------------------
ax.text(0.1, 6.5, "DATA", fontsize=11, weight="bold")
ax.text(0.1, 4.5, "MODEL", fontsize=11, weight="bold")
ax.text(0.1, 2.5, "DECISION", fontsize=11, weight="bold")
ax.text(0.1, 0.5, "OUTPUT", fontsize=11, weight="bold")

# -----------------------------
# LIMITS
# -----------------------------
ax.set_xlim(0, 1)
ax.set_ylim(-1, 8)

plt.tight_layout()
plt.savefig("outputs/architecture_diagram.png", dpi=300)

print("Architecture diagram created")
