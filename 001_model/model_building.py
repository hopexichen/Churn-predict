import pickle

import pandas as pd
import numpy as np


pd.set_option('display.max_columns', None)

from sklearn import metrics
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from xgboost.sklearn import XGBClassifier


df = pd.read_csv("churn.csv")

# Generate features and target
features = ['avg_transaction_value', 'points_in_wallet', 'avg_frequency_login_days', 'membership_category', 'feedback']
target = 'churn_risk_score'

X = df[features]
y = df[target]

# Split DataFrame into train data and test data
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                      y, 
                                                      test_size=0.2, 
                                                      random_state=2022)


# step1 transform
class Data_Transformer():
    
    def fit(self, X, y=None):
        df = pd.DataFrame()
        df['avg_transaction_value'] = X['avg_transaction_value']        
        df['points_in_wallet'] = X['points_in_wallet'].map(lambda x: abs(float(x)))     
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
X_test_s1 = dtf.transform(X_test)


# step2 transform
transformer_ordinal = ColumnTransformer(transformers = [
    ('tn1', MinMaxScaler(), ['avg_transaction_value', 'points_in_wallet', 'avg_frequency_login_days']),
    ('ordinal',OrdinalEncoder(categories=[['No Membership', 'Basic Membership', 'Silver Membership', 'Gold Membership', 'Platinum Membership', 'Premium Membership']]),['membership_category']),
],remainder='passthrough')

column_names = ['avg_transaction_value', 'points_in_wallet', 'avg_frequency_login_days', 'membership_category', 'feedback'] 

X_train_s2 = transformer_ordinal.fit_transform(X_train_s1)
X_train_s2 = pd.DataFrame(X_train_s2, columns = column_names)
X_test_s2 = transformer_ordinal.transform(X_test_s1)
X_test_s2 = pd.DataFrame(X_test_s2, columns = column_names)


# step3 transform
transformer_onehot = ColumnTransformer(transformers = [
    ('feedback',OneHotEncoder(sparse=False,handle_unknown='ignore'),['feedback']),
],remainder='passthrough')

column_names_final = ['feedback_No_reason_specified','feedback_Poor_Customer_Service','feedback_Poor_Product_Quality','feedback_Poor_Website','feedback_Products_always_in_Stock','feedback_Quality_Customer_Care','feedback_Reasonable_Price','feedback_Too_many_ads','feedback_User_Friendly_Website', 'avg_transaction_value', 'points_in_wallet', 'avg_frequency_login_days', 'membership_category']

X_train_s3 = transformer_onehot.fit_transform(X_train_s2)
X_train_s3 = pd.DataFrame(X_train_s3, columns=column_names_final)
X_test_s3 = transformer_onehot.transform(X_test_s2)
X_test_s3 = pd.DataFrame(X_test_s3, columns=column_names_final)



model = GradientBoostingClassifier(random_state=2022, learning_rate=0.1, max_depth=3, min_samples_leaf=3, min_samples_split=2, n_estimators=200, max_features='sqrt')

model.fit(X_train_s3, y_train)


with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
