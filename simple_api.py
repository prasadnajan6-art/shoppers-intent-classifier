# ================================================================
# SIMPLE FLASK API
# This connects our trained model to the webpage
# ================================================================
# HOW TO RUN:
#   py simple_api.py
#
# INSTALL FIRST:
#   py -m pip install flask flask-cors joblib scikit-learn pandas
# ================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Create the Flask app
app = Flask(__name__)
CORS(app)  # This allows the webpage to talk to this API

# Load our saved model
print("Loading model...")
model = joblib.load('simple_model.pkl')
print("✅ Model loaded!")

# These are all the months and their number codes
# (must match what we used during training)
MONTH_MAP = {
    'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3,
    'May': 4, 'Jun': 5, 'Jul': 6, 'Aug': 7,
    'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11
}

VISITOR_MAP = {
    'New_Visitor': 0,
    'Other': 1,
    'Returning_Visitor': 2
}

# This is the prediction endpoint
# When the webpage sends data here, we return a prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data sent from the webpage
        d = request.get_json()

        # Build the input in the same order as training
        features = [[
            float(d['Administrative']),
            float(d['Administrative_Duration']),
            float(d['Informational']),
            float(d['Informational_Duration']),
            float(d['ProductRelated']),
            float(d['ProductRelated_Duration']),
            float(d['BounceRates']),
            float(d['ExitRates']),
            float(d['PageValues']),
            float(d['SpecialDay']),
            MONTH_MAP.get(d['Month'], 9),
            int(d['OperatingSystems']),
            int(d['Browser']),
            int(d['Region']),
            int(d['TrafficType']),
            VISITOR_MAP.get(d['VisitorType'], 2),
            int(d['Weekend']),
        ]]

        # Make prediction
        result      = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        # Send result back to webpage
        return jsonify({
            'prediction'          : int(result),
            'label'               : 'Purchase' if result == 1 else 'No Purchase',
            'confidence'          : round(float(max(probability)) * 100, 1),
            'purchase_probability': round(float(probability[1]) * 100, 1)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/')
def home():
    return "✅ Shopper Prediction API is running!"


if __name__ == '__main__':
    print("\n🚀 API running at http://localhost:5000")
    print("   Keep this terminal open while using the webpage!\n")
    app.run(debug=True, port=5000)
