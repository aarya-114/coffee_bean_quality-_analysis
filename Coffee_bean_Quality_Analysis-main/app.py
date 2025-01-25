from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from tensorflow import keras
import numpy as np
import os
import logging
import traceback
import pickle
import requests
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app.config['UPLOAD_FOLDER'] = 'temp_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Print current working directory and list files
app.logger.info(f"Current working directory: {os.getcwd()}")
app.logger.info(f"Files in current directory: {os.listdir()}")
app.logger.info(f"Files in templates directory: {os.listdir('templates')}")

# Global variables for model and scaler
keras_model = None
scaler = None

def load_model_and_scaler():
    global keras_model, scaler
    
    model_path = 'coffee_quality_model1.keras'
    if os.path.exists(model_path):
        keras_model = keras.models.load_model(model_path)
        app.logger.info("Keras model loaded successfully")
    else:
        app.logger.error(f"Error: Keras model file not found at {model_path}")

    scaler_path = 'coffee_quality_scaler1.pkl'
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        app.logger.info("Scaler loaded successfully")
    else:
        app.logger.error(f"Error: Scaler file not found at {scaler_path}")

load_model_and_scaler()

ROBOFLOW_API_KEY = 'qsnHim4X5wfVW73yoZBC'

@app.route('/')
def home():
    app.logger.info("Home route accessed")
    return render_template('home.html')

@app.route('/text_input')
def text_input():
    app.logger.info("Text input route accessed")
    return render_template('text_input.html')

@app.route('/image_input')
def image_input():
    app.logger.info("Image input route accessed")
    return render_template('image_input.html')

@app.route('/predict_text', methods=['POST'])
def predict_text():
    app.logger.info("Predict text route accessed")
    try:
        if keras_model is None or scaler is None:
            raise ValueError("Model or scaler not loaded. Please check the files and restart the application.")
        
        # Parse incoming JSON data
        data = request.get_json()
        app.logger.debug(f"Form data: {data}")

        # Extract input data
        input_data = [
            data['Aroma'],     # Already numeric due to the mapCategoryToNumeric function in JS
            data['Flavor'],
            data['Aftertaste'],
            data['Acidity'],
            data['Body'],
            float(data['Moisture']),  # Keep moisture as a float
        ]
        app.logger.debug(f"Input data: {input_data}")
        
        # You should scale all input data, not just moisture
        scaled_input = scaler.transform([input_data])
        app.logger.debug(f"Scaled input: {scaled_input}")
        
        # Make prediction
        prediction = keras_model.predict(scaled_input)
        app.logger.debug(f"Prediction: {prediction}")
        
        return jsonify({'prediction': float(prediction[0][0])})
    except Exception as e:
        app.logger.error(f"Error in predict_text: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/predict_image', methods=['POST'])
def predict_image():
    app.logger.info("Predict image route accessed")
    try:
        data = request.json
        if 'image' not in data:
            raise ValueError("No image data provided")

        base64_image = data['image']
        
        app.logger.info("Sending request to Roboflow API")
        response = requests.post(
            'https://classify.roboflow.com/coffee-defect/1',
            params={
                'api_key': ROBOFLOW_API_KEY
            },
            data=base64_image,
            headers={
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        )

        app.logger.info(f"Roboflow API response status: {response.status_code}")
        app.logger.info(f"Roboflow API response content: {response.text}")

        if response.status_code == 200:
            app.logger.debug(f"Prediction response: {response.json()}")
            return jsonify(response.json())
        else:
            error_message = f"Roboflow API error: {response.status_code} - {response.text}"
            app.logger.error(error_message)
            return jsonify({'error': error_message}), response.status_code
    except Exception as e:
        error_message = f"Error in predict_image: {str(e)}"
        app.logger.error(error_message)
        app.logger.error(traceback.format_exc())
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)
