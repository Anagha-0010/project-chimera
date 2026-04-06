import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# 1. Load the results
df = pd.read_csv("roc_results.csv")

# 2. Calculate AUC again for the legend
roc_auc = auc(df['fpr'], df['tpr'])

# 3. Create the Plot
plt.figure(figsize=(8, 8))
plt.plot(df['fpr'], df['tpr'], color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal line

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Clean prompts flagged as malicious)', fontsize=12)
plt.ylabel('True Positive Rate (Poisoned prompts correctly caught)', fontsize=12)
plt.title('Project Chimera: ROC Curve for Layer 19 Probe', fontsize=15)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# 4. Save the plot
plt.savefig("chimera_roc_curve.png", dpi=300)
print("✅ ROC Curve saved to chimera_roc_curve.png")