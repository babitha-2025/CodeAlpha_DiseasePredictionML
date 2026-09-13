import pandas as pd
import joblib

# ==========================================
# 1. Load trained model
# ==========================================

model = joblib.load("models/heart_disease_xgboost.pkl")

# ==========================================
# 2. Create sample patient
# ==========================================

patient = pd.DataFrame([{
    "age": 55,
    "sex": 1,
    "cp": 4,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 1,
    "oldpeak": 2.0,
    "slope": 2,
    "ca": 1,
    "thal": 3
}])

# ==========================================
# 3. Make prediction
# ==========================================

prediction = model.predict(patient)[0]

probability = model.predict_proba(patient)[0]

# ==========================================
# 4. Display result
# ==========================================

print("========== HEART DISEASE PREDICTION ==========")

if prediction == 1:
    print("Prediction: Heart Disease")
else:
    print("Prediction: No Heart Disease")

print("\nProbability of No Disease:", round(probability[0] * 100, 2), "%")
print("Probability of Disease:", round(probability[1] * 100, 2), "%")