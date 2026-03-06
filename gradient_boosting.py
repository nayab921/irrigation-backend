import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

print("Starting Gradient Boosting Pipeline...")
# 1. Load & Split
df = pd.read_csv('Pakistan_Smart_Irrigation_Dataset.csv')
X = df[['District', 'Crop', 'Soil', 'Area_Hectares', 'Motor_HP']]
y = df['Total_Liters_Used']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Pipeline
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['District', 'Crop', 'Soil'])],
    remainder='passthrough'
)
model = Pipeline(steps=[('preprocessor', preprocessor), 
                        ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))])

# 3. Train
model.fit(X_train, y_train)

# 4. Results
y_pred = model.predict(X_test)
print("\n--- MODEL REPORT (Gradient Boosting) ---")
print(f"R² Score: {r2_score(y_test, y_pred)*100:.2f}%")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

# 5. Graph
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.title('Gradient Boosting: Predictions vs Reality')
plt.savefig('graph_gb_results.png')
print("Graph saved as 'graph_gb_results.png'")


print("\nChecking Overfitting...")
train_accuracy = model.score(X_train, y_train) * 100
test_accuracy = r2_score(y_test, y_pred) * 100

print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Testing Accuracy:  {test_accuracy:.2f}%")
print(f"Gap:               {train_accuracy - test_accuracy:.2f}%")

joblib.dump(model, 'final_model_gb.pkl')