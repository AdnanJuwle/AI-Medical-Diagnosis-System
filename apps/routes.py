# app/routes.py
from flask import Flask, request, jsonify
from model_loader import diabetes_model, lung_cancer_model

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    disease_type = data.get('disease_type')
    features = data.get('features')

    if disease_type == 'diabetes':
        model = diabetes_model
    elif disease_type == 'lung_cancer':
        model = lung_cancer_model
    else:
        return jsonify({'error': 'Invalid disease type'}), 400

    # Convert features to a DataFrame or appropriate format
    # Assuming features is a list or array
    import pandas as pd
    features_df = pd.DataFrame([features])

    # Make prediction
    prediction = model.predict(features_df)
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
