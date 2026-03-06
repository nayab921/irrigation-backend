from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app) # Security permit for frontend to access API

# 1. Gradient Boosting Model Load 
model = joblib.load('final_model_gb.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Input Data ko DataFrame
        input_df = pd.DataFrame([{
            'District': data['District'],
            'Crop': data['Crop'],
            'Soil': data['Soil'],
            'Area_Hectares': float(data['Area_Hectares']),
            'Motor_HP': float(data['Motor_HP'])
        }])

        # 2. ML Model Water Requirement (Liters)
        liters = float(model.predict(input_df)[0])

        # 3. Irrigation Hours Calculation
        # Logic: Liters / (MotorHP * 7500) -> 7500 standard flow factor
        motor_hp = float(data['Motor_HP'])
        hours = liters / (motor_hp * 7500)

        # 4. Result JSON format
        return jsonify({
            'water_liters': round(liters, 2),
            'irrigation_hours': round(hours, 2),
            'status': 'success'
        })

    except Exception as e:
        # frontend error
        return jsonify({'error': str(e), 'status': 'failed'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)