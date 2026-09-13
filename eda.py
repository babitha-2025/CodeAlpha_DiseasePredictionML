import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/heart_disease_cleaned.csv")

# Count patients with and without heart disease
target_counts = df["target"].value_counts().sort_index()

# Create bar chart
plt.figure(figsize=(7, 5))

plt.bar(
    ["No Disease", "Disease"],
    target_counts.values
)

plt.title("Heart Disease Distribution")
plt.xlabel("Condition")
plt.ylabel("Number of Patients")

# Add values on top of bars
for i, value in enumerate(target_counts.values):
    plt.text(i, value + 3, str(value), ha="center")

plt.tight_layout()

# Save the graph
plt.savefig("data/heart_disease_distribution.png", dpi=300)

plt.show()

print("EDA graph created successfully!")
print("\nPatient distribution:")
print(target_counts)