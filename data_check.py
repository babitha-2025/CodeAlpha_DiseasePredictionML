import pandas as pd

columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

# Load original dataset
df = pd.read_csv(
    "data/processed.cleveland.data",
    header=None,
    names=columns,
    na_values="?"
)

# Fill missing values
df["ca"] = df["ca"].fillna(df["ca"].median())
df["thal"] = df["thal"].fillna(df["thal"].median())

# Convert target to binary
# 0 = No disease
# 1 = Disease
df["target"] = (df["target"] > 0).astype(int)

# Save cleaned dataset
df.to_csv("data/heart_disease_cleaned.csv", index=False)

print("Cleaned dataset saved successfully!")
print("File: data/heart_disease_cleaned.csv")
print("Shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum().sum())