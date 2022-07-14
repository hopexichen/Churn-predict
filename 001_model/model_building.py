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
#from sklearn.ensemble import RandomForestClassifier
#from xgboost.sklearn import XGBClassifier


df = pd.read_csv("churn.csv")

# Generate features and target
features = ['avg_transaction_value']
target = 'churn_risk_score'

X = df[features]
y = df[target]
# X['points_in_wallet'] = abs(X['points_in_wallet'])

# Split DataFrame into train data and test data
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                      y, 
                                                      test_size=0.2, 
                                                      random_state=2022)

# X_train['points_in_wallet'].fillna(X_train['points_in_wallet'].median, inplace=True)

model = LogisticRegression()

model.fit(X_train, y_train)


with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
