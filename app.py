from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model, scaler, and label encoder
model = pickle.load(open('alzheimers_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# Ensure the selected features match between training and application
FEATURES = ['BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
            'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
            'MMSE', 'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = {feature: request.form[feature] for feature in FEATURES}
        input_features = [float(data[feature]) for feature in FEATURES]

        # Convert input data into DataFrame
        input_df = pd.DataFrame([input_features], columns=FEATURES)

        # Scale the input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_label = 'Positive' if prediction[0] == 1 else 'Negative'

        return render_template('index.html', prediction=f'Alzheimer Prediction: {prediction_label}')
    
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

# API endpoint for JSON-based prediction
@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        # Get data from request
        data = request.get_json()

        # Convert data to list of features
        input_features = [float(data[feature]) for feature in FEATURES]

        # Convert input data into DataFrame
        input_df = pd.DataFrame([input_features], columns=FEATURES)

        # Scale the input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)

        # Determine result based on prediction
        prediction_label = 'Positive' if prediction[0] == 1 else 'Negative'

        return jsonify({"prediction": prediction_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
