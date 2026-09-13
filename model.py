import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier


# ==========================================
# 1. Load cleaned dataset
# ==========================================

df = pd.read_csv("data/heart_disease_cleaned.csv")


# ==========================================
# 2. Separate features and target
# ==========================================

X = df.drop("target", axis=1)
y = df["target"]


# ==========================================
# 3. Define categorical columns
# ==========================================

categorical_features = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]


# ==========================================
# 4. Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ==========================================
# 5. Create XGBoost model
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss"
            )
        )
    ]
)


# ==========================================
# 6. Split data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 7. Train model
# ==========================================

model.fit(X_train, y_train)

print("\nXGBoost training completed successfully!")


# ==========================================
# 8. Make predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 9. Evaluate model
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== XGBOOST RESULTS ==========")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Disease", "Disease"]
    )
)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# ==========================================
# 10. Create confusion matrix graph
# ==========================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Disease"]
)

display.plot()

plt.title("XGBoost Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "data/xgboost_confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix graph saved successfully!")


# ==========================================
# 11. Save trained model
# ==========================================

joblib.dump(
    model,
    "models/heart_disease_xgboost.pkl"
)

print("Trained XGBoost model saved successfully!")
print("File: models/heart_disease_xgboost.pkl")

# ==========================================
# 12. Feature importance
# ==========================================

# Get the trained XGBoost classifier
xgb_model = model.named_steps["classifier"]

# Get feature names after preprocessing
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# Get importance values
importance = xgb_model.feature_importances_

# Create a DataFrame
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# Sort by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== TOP FEATURE IMPORTANCES ==========")
print(feature_importance.head(10))

# Plot top 10 features
plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"].head(10)[::-1],
    feature_importance["Importance"].head(10)[::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")

plt.tight_layout()

plt.savefig(
    "data/xgboost_feature_importance.png",
    dpi=300
)

plt.show()

print("\nFeature importance graph saved successfully!")