from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)
model = load("real_estate_model.joblib")
pipeline = load("real_estate_pipeline.joblib")
labels = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

@app.route("/", methods=["GET"])
def home():
    return render_template("form.html", labels=labels, price=None)

@app.route('/predict', methods=['POST'])
def predictt():
    try:
        raw_features = request.form.getlist('features[]')

        if not raw_features:
            raise ValueError("No input features provided.")

        # Convert "yes"/"no" to 1/0 for binary inputs like CHAS
        for idx, val in enumerate(raw_features):
            if val.lower() == "yes":
                raw_features[idx] = 1
            elif val.lower() == "no":
                raw_features[idx] = 0

        features = list(map(float, raw_features))

        # Validate input length
        if len(features) != len(labels):
            raise ValueError("All fields are required.")

        # Convert to DataFrame for compatibility with pipeline
        import pandas as pd
        input_df = pd.DataFrame([features], columns=labels)

        # Transform using the saved pipeline
        prepared_input = pipeline.transform(input_df)

        # Predict
        prediction = model.predict(prepared_input)

        return render_template('form.html', labels=labels, price=prediction[0], error=None)

    except Exception as e:
        return render_template('form.html', labels=labels, price=None, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
