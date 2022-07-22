from flask import Flask, jsonify, request, render_template
import json
import pandas as pd
import numpy as np
import pickle


from sklearn import metrics
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, log_loss
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, plot_roc_curve, plot_precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        avg_transaction_value = request.form["avg_transaction_value"]
        points_in_wallet = request.form["points_in_wallet"]
        avg_frequency_login_days = request.form["avg_frequency_login_days"]
        
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
        
        
        operator = request.form["operator"]
        if operator == "Reasonable Price":
            feedback = "feedback_Reasonable_Price"
        elif operator == "Quality Customer Care":
            feedback = "feedback_Quality_Customer_Care" 
        elif operator == "User Friendly Website":
            feedback = "feedback_User_Friendly_Website" 
        elif operator == "Products always in Stock":
            feedback = "feedback_Products_always_in_Stock" 
        elif operator == "Poor Product Quality":
            feedback = "feedback_Poor_Product_Quality" 
        elif operator == "Poor Customer Service":
            feedback = "feedback_Poor_Customer_Service" 
        elif operator == "Poor Website":
            feedback = "feedback_Poor_Website" 
        elif operator == "Too many ads":
            feedback = "feedback_Too_many_ads" 
        else:
            feedback = "feedback_No_reason_specified" 
        
        
        
        column_names = ['avg_transaction_value', 'points_in_wallet', 'avg_frequency_login_days', 'membership_category', 'feedback']      
        X_input = pd.DataFrame([[avg_transaction_value, points_in_wallet, avg_frequency_login_days, membership_category, feedback]], columns=column_names)
        
        
        df = pd.read_csv("churn.csv")
        features = ['avg_transaction_value', 'points_in_wallet', 'avg_frequency_login_days', 'membership_category', 'feedback']
        target = 'churn_risk_score'
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                      y, 
                                                      test_size=0.2, 
                                                      random_state=2022)

        class Data_Transformer():
    
            def fit(self, X, y=None):
                df = pd.DataFrame()
                df['avg_transaction_value'] = X['avg_transaction_value']        
                df['points_in_wallet'] =  X['points_in_wallet'].map(lambda x: abs(float(x)))        
                df['avg_frequency_login_days'] = X['avg_frequency_login_days'].map(lambda x: abs(float(x)) if x != 'Error' else np.nan)
        
                self.median = df.median()
        
            def transform(self, X, y=None):
                df = pd.DataFrame()
                df1 = pd.DataFrame()
                df['avg_transaction_value'] = X['avg_transaction_value']        
                df['points_in_wallet'] = X['points_in_wallet'].map(lambda x: abs(float(x)))        
                df['avg_frequency_login_days'] = X['avg_frequency_login_days'].map(lambda x: abs(float(x)) if x != 'Error' else np.nan)
        
                df.fillna(self.median, inplace=True)
        
                # put in selected categorical features 
                df1['membership_category'] = X['membership_category']
                df1['feedback'] = X['feedback']
        
                df1.fillna('unknown', inplace=True)
                df = pd.concat([df, df1], axis=1)
    
                return df
    
            def fit_transform(self, X, y=None):
                self.fit(X, y=None)
                return self.transform(X, y=None)
    
        
        dtf = Data_Transformer()
        X_train_s1 = dtf.fit_transform(X_train)
        X_s1 = dtf.transform(X_input)
       

        transformer_ordinal = ColumnTransformer(transformers = [
    ('tn1', MinMaxScaler(), ['avg_transaction_value', 'points_in_wallet', 'avg_frequency_login_days']),
    ('ordinal',OrdinalEncoder(categories=[['No Membership', 'Basic Membership', 'Silver Membership', 'Gold Membership', 'Platinum Membership', 'Premium Membership']]),['membership_category']),
],remainder='passthrough')

        X_train_s2 = transformer_ordinal.fit_transform(X_train_s1)
        X_train_s2 = pd.DataFrame(X_train_s2, columns = column_names)
        X_s2 = transformer_ordinal.transform(X_s1)
        X_s2 = pd.DataFrame(X_s2, columns = column_names)
        
        transformer_onehot = ColumnTransformer(transformers = [
    ('feedback',OneHotEncoder(sparse=False,handle_unknown='ignore'),['feedback']),
],remainder='passthrough')

        column_names_final = ['feedback_No_reason_specified','feedback_Poor_Customer_Service','feedback_Poor_Product_Quality','feedback_Poor_Website','feedback_Products_always_in_Stock','feedback_Quality_Customer_Care','feedback_Reasonable_Price','feedback_Too_many_ads','feedback_User_Friendly_Website', 'avg_transaction_value', 'points_in_wallet', 'avg_frequency_login_days', 'membership_category']

        X_train_s3 = transformer_onehot.fit_transform(X_train_s2)
        X_train_s3 = pd.DataFrame(X_train_s3, columns=column_names_final)
        X_s3 = transformer_onehot.transform(X_s2)
        X_s3 = pd.DataFrame(X_s3, columns=column_names_final)


        #X = np.array([[float(avg_transaction_value)]])
        pred = model.predict_proba(X_s3)[0][1]
    
    return render_template("index.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
