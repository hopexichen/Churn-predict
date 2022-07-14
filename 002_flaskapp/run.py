from flask import Flask, jsonify, request, render_template
import json
import pandas as pd
import numpy as np
import pickle


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        avg_transaction_value = request.form["avg_transaction_value"]
        points_in_wallet = request.form["points_in_wallet"]
        operator = request.form["operator"]
        if operator == "Premium Membership":
            membership_category = "Premium Membership"
        elif operator == "Platinum Membership":
            membership_category = "Platinum Membership" 
        elif operator == "Gold Membership":
            membership_category = "Gold Membership" 
        elif operator == "Silver Membership":
            membership_category = "Silver Membership" 
        elif operator == "Basic Membership":
            membership_category = "Basic Membership" 
        else:
            membership_category = "No Membership" 
             
             
        #X = pd.DataFrame([[float(avg_transaction_value), float(points_in_wallet), membership_category]])
        
        X = np.array([[float(avg_transaction_value)]])
        pred = model.predict_proba(X)[0][1]
    
    return render_template("index.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
