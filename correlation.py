import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/heart_disease_cleaned.csv")

# Calculate correlations
correlation = df.corr()

# Create heatmap
plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Heart Disease Dataset")
plt.tight_layout()

# Save the graph
plt.savefig("data/correlation_heatmap.png", dpi=300)

plt.show()

print("Correlation heatmap created successfully!")