import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the model, scaler, and feature names
with open('california_housing_rf.pkl', 'rb') as f:
    deployment_assets = pickle.load(f)
    model = deployment_assets['model']
    scaler = deployment_assets['scaler']
    feature_names = deployment_assets['feature_names']

@app.route('/')
def home():
    return render_template('home.html')  # Make sure this file exists

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        input_array = np.array([data[col] for col in feature_names]).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)