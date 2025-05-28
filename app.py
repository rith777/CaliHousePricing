import pickle
import json
from flask import Flask, request, jsonify, render_template, url_for,app
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load both model AND scaler (updated pickle file)
with open('california_housing_rf.pkl', 'rb') as f:
    deployment_assets = pickle.load(f)
    model = deployment_assets['model']
    scaler = deployment_assets['scaler']  # Now available
    feature_names = deployment_assets['feature_names']  # For validation

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get and validate input
        data = request.json['data']
        
        # Convert to properly ordered array
        input_array = np.array([data[col] for col in feature_names]).reshape(1, -1)
        
        # Transform and predict
        scaled_input = scaler.transform(input_array)  # Now using the loaded scaler
        output = model.predict(scaled_input)
        
        return jsonify({
            'prediction': float(output[0]),
            'status': 'success',
            'units': 'Price in $100,000s'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed',
            'required_features': feature_names
        }), 400
@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template('home.html', prediction_text='House Price in $100,000s: {}'.format(output))
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)