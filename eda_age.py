import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/heart_disease_cleaned.csv")

# Create box plot
plt.figure(figsize=(7, 5))

df.boxplot(
    column="age",
    by="target"
)

plt.title("Age Distribution by Heart Disease Status")
plt.suptitle("")
plt.xlabel("Heart Disease Status")
plt.ylabel("Age")

plt.xticks([1, 2], ["No Disease", "Disease"])

plt.tight_layout()

# Save graph
plt.savefig("data/age_vs_heart_disease.png", dpi=300)

plt.show()

print("Age analysis graph created successfully!")