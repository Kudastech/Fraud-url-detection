# app.py

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
import warnings
from feature import FeatureExtraction

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the model
try:
    gbc = joblib.load("pickle/model.pkl")
except Exception as e:
    gbc = None
    print(f"Error loading model: {e}")

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    pred = ""
    y_pro_non_phishing = -1

    if request.method == "POST":
        url = request.form["url"]
        try:
            obj = FeatureExtraction(url)
            x = np.array(obj.getFeaturesList()).reshape(1, 30)

            if gbc:
                y_pred = gbc.predict(x)[0]
                y_pro_phishing = gbc.predict_proba(x)[0, 0]
                y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

                pred = f"It is {y_pro_phishing*100:.2f} % safe to go"
            else:
                pred = "Model not loaded. Please check the model file."

        except Exception as e:
            pred = f"Error during prediction: {e}"

        return render_template('index.html', xx=round(y_pro_non_phishing, 2), url=url, pred=pred)
    
    return render_template("index.html", xx=-1)

if __name__ == "__main__":
    app.run(debug=True)
