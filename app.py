from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load models
models = {
    'lr_model': joblib.load('lr_model.pkl'),
    'mlp': joblib.load('mlp.pkl'),
    'stacking_model': joblib.load('stacking_model.pkl')
}


@app.route('/predict', methods=['GET'])
def predict():
    # Retrieve query parameters
    model_choice = request.args.get('model')
    input_data = request.args.getlist('inputs', type=float)
    years_to_predict = request.args.get('years', default=5, type=int)

    if model_choice not in models:
        return jsonify({'error': f'Model not found. Choose from {", ".join(models.keys())}.'}), 400

    # Prepare the input data
    processed_data = np.array(input_data).reshape(1, -1)

    # List to hold predictions for each year
    future_predictions = []

    # Generate predictions for each year
    for year in range(years_to_predict):
        prediction = models[model_choice].predict(processed_data)
        future_predictions.append({f'Year_{year + 1}': prediction.tolist()})
        # If necessary, update processed_data for subsequent predictions

    # Return predictions
    return jsonify({'predictions': future_predictions})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
