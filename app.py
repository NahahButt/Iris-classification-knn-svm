
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'iris_svm_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'iris_scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    return "Iris Prediction API. Use /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded'}), 500

    try:
        data = request.get_json(force=True)
        
        # Extract features from the JSON payload
        # Ensure the order of features matches the training data
        features = [
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]
        
        # Convert to DataFrame to maintain feature names for scaling if needed
        # For StandardScaler, a list of lists is sufficient for a single sample
        input_df = pd.DataFrame([features], columns=[
            'sepal length (cm)', 
            'sepal width (cm)', 
            'petal length (cm)', 
            'petal width (cm)'
        ])

        # Scale the input features
        scaled_features = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        predicted_class = int(prediction[0])

        # Map class labels to species names (assuming original Iris dataset mapping)
        species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        predicted_species = species_map.get(predicted_class, 'Unknown')

        return jsonify({
            'prediction': predicted_class,
            'species': predicted_species
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # In Colab, you might need to run on all interfaces to access it externally
    # However, for local testing, default is fine.
    # For deployment, consider using a production-ready WSGI server like Gunicorn
    app.run(host='0.0.0.0', port=5000)
