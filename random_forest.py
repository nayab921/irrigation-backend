import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# 1. SETUP
print("Starting Random Forest Pipeline...")
df = pd.read_csv('Pakistan_Smart_Irrigation_Dataset.csv')
X = df[['District', 'Crop', 'Soil', 'Area_Hectares', 'Motor_HP']]
y = df['Total_Liters_Used']

# 2. PREPROCESSING
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['District', 'Crop', 'Soil'])
    ], remainder='passthrough'
)

# 3. TRAIN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Pipeline(steps=[('preprocessor', preprocessor), 
                        ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
model.fit(X_train, y_train)

# 4. EVALUATION
y_pred = model.predict(X_test)
print("\n--- MODEL REPORT (Random Forest) ---")
print(f"R² Score (Accuracy): {r2_score(y_test, y_pred)*100:.2f}%")
print(f"MAE (Average Error): {mean_absolute_error(y_test, y_pred):.2f}")


# Hack to get feature names after OneHotEncoding
rf_model = model.named_steps['model']
preprocessor_step = model.named_steps['preprocessor']
cat_features = preprocessor_step.named_transformers_['cat'].get_feature_names_out(['District', 'Crop', 'Soil'])
all_features = np.concatenate([cat_features, ['Area_Hectares', 'Motor_HP']])

importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:] # Top 10 features

plt.figure(figsize=(10, 6))
plt.title('Top 10 Important Factors for Water Usage')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [all_features[i] for i in indices])
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('graph_rf_importance.png')
print("Feature Importance Graph saved as 'graph_rf_importance.png'")


# --- OVERFITTING CHECK ---
print("\nChecking for Overfitting...")
train_score = model.score(X_train, y_train) * 100
test_score = model.score(X_test, y_test) * 100

print(f"Training Score : {train_score:.2f}%")
print(f"Testing Score :  {test_score:.2f}%")
print(f"Gap :            {train_score - test_score:.2f}%")

if (train_score - test_score) < 5:
    print("Result: Model Safe")
else:
    print("Result: Model Overfit")

# 6. SAVE
joblib.dump(model, 'final_model_rf.pkl')
print("Model saved!")