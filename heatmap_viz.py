import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv("heatmap_data.csv")

# 2. Pivot for the Heatmap (Layers vs. Trigger Type)
# We average across domains to see the "Global Signature"
pivot_df = df.groupby(["TriggerType", "Layer"])["Prob"].mean().unstack()

# 3. Create the Visualization
plt.figure(figsize=(15, 6))
sns.heatmap(
    pivot_df, 
    annot=False,      # Set to True if you want numbers in the boxes
    cmap="magma",     # 'magma' or 'viridis' are great for visibility
    cbar_kws={'label': 'Malicious Probability (Token: "import")'}
)

plt.title("Project Chimera: Mechanistic 'Sleeper' Activation Heatmap", fontsize=15)
plt.xlabel("Transformer Layer (0-25)", fontsize=12)
plt.ylabel("Trigger Variation", fontsize=12)

# Save the file
plt.tight_layout()
plt.savefig("chimera_heatmap.png", dpi=300)
print("✅ Static heatmap saved to chimera_heatmap.png")