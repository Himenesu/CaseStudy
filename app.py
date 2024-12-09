from flask import Flask, request, jsonify
import joblib

# Initialize Flask application
app = Flask(__name__)

# Load pre-trained models (ensure that the models are saved as .pkl files)
lr_model = joblib.load('lr_model.pkl')  # Linear Regression Model
mlp_model = joblib.load('mlp_model.pkl')  # Neural Network Model
stacking_model = joblib.load('stacking_model.pkl')  # Stacking Regressor Model

@app.route('/predict', methods=['POST'])
def predict():
    # Get the features from the POST request (expects JSON format with 'features' key)
    data = request.get_json(force=True)
    features = [data['features']]  # This should be in the form of a list, e.g., {'features': [value1, value2, ...]}

    # Make predictions using the pre-trained models
    lr_pred = lr_model.predict(features)
    mlp_pred = mlp_model.predict(features)
    stacking_pred = stacking_model.predict(features)

    # Return the predictions in a JSON response
    response = {
        'Linear Regression Prediction': lr_pred.tolist(),
        'Neural Network Prediction': mlp_pred.tolist(),
        'Stacking Regressor Prediction': stacking_pred.tolist()
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
