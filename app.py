from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load pre-trained models
lr_model = joblib.load('lr_model.pkl')
mlp_model = joblib.load('mlp.pkl')
stacking_model = joblib.load('stacking_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from request
    data = request.json
    features = np.array(data['features']).reshape(1, -1)

    # Make predictions
    prediction_lr = lr_model.predict(features)
    prediction_mlp = mlp_model.predict(features)
    prediction_stacking = stacking_model.predict(features)

    # Return predictions as JSON
    return jsonify({
        'Linear Regression Prediction': prediction_lr.tolist(),
        'MLP Prediction': prediction_mlp.tolist(),
        'Stacking Regressor Prediction': prediction_stacking.tolist()
    })

@app.route('/')
def home():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

