from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app) # Security permit

# Model Load karein
model = joblib.load('irrigation_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # 1. Input Data
        input_df = pd.DataFrame([{
            'District': data['District'],
            'Crop': data['Crop'],
            'Soil': data['Soil'],
            'Area_Hectares': float(data['Area_Hectares']),
            'Motor_HP': float(data['Motor_HP'])
        }])

        # 2. Prediction (Liters)
        liters = model.predict(input_df)[0]

        # 3. Calculation (Hours)
        # Logic: Liters / (MotorHP * 7500) -> 7500 aik standard flow factor hai
        hours = liters / (float(data['Motor_HP']) * 7500)

        # 4. Result Wapis Bhejein
        return jsonify({
            'water_liters': round(liters, 2),
            'irrigation_hours': round(hours, 2),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)