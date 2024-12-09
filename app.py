from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load models
models = {
    'model1': joblib.load('model1.pkl'),
    'model2': joblib.load('model2.pkl'),
    'model3': joblib.load('model3.pkl')
}


@app.route('/predict', methods=['POST'])
def predict():
    # Example input: {'model': 'model1', 'inputs': [list_of_features]}
    data = request.json
    model_choice = data.get('model')
    input_data = data.get('inputs')

    if model_choice not in models:
        return jsonify({'error': 'Model not found. Choose from model1, model2, model3.'}), 400

    # Process the input data
    processed_data = np.array(input_data).reshape(1, -1)

    # Use chosen model for prediction
    prediction = models[model_choice].predict(processed_data)

    # Return the prediction result
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
