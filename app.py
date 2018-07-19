import numpy as np
import pandas as pd
import dill
import json
import sklearn
import lime
import lime.lime_tabular
from urllib import request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from flask import Flask, request
app = Flask(__name__)

# read the data
print("Reading the data")
#data = data = pd.read_csv("preprocessed_LC.csv", na_filter=False)
data = dill.load(open('data', 'rb'))
print("Reading...done")

# preprocessing
print("Data Preprocessing started")
feature_names = list(data.columns[:-1])
labels = data.status.values
data = data.drop(['status'], axis=1)

# get class names
class_names = ['Rejected', 'Approved']

# get the categorical features
categorical_features = [1, 4]

# preprocessing emp_length
encoding = {
    'emp_length': {
        '< 1 year': 'less_than_1_year',
        '1 year': '1_year',
        '10+ years': '10_or_more_years',
        '6 years': '6_years',
        '7 years': '7_years',
        '5 years': '5_years',
        '3 years': '3_years',
        '2 years': '2_years',
        '4 years':'4_years',
        '9 years': '9_years',
        '8 years': '8_years',
        'n/a' : 'Not Specified'
    }
}

data.replace(encoding, inplace=True)
data = data.values

# create dictionary of categorical names
categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:, feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_

categorical_names[1] = np.array(categorical_names[1], dtype=str)
categorical_names[4] = np.array(['1_year', '10_or_more_years', '2_years', '3_years', '4_years',
                                 '5_years', '6_years', '7_years', '8_years', '9_years',
                                 'less_than_1_year', 'Not Specified'])
    
data = data.astype(float)
print("Data Preprocessing...done")

# one hot encoding
encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_features)

# train/test split
print("Creating Train/Test Split datasets")
np.random.seed(42)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, test_size=0.20)
print("Creating Train/Test Split datasets...done")


# encode training set
encoder.fit(data)
encoded_train = encoder.transform(train)

# build the model
print("Loading trained model")
model = dill.load(open('model', 'rb'))
print("Done")

predict_fn = lambda x: model.predict_proba(encoder.transform(x)).astype(float)


# Explaining prediction
explainer = dill.load(open('explainer', 'rb'))

# get the explaination for instance
@app.route("/application/", methods=['POST', 'GET'])
def get_reason_codes():

	# Loan_Amount = loan_amount
	# Loan_Purpose = loan_purpose
	# Credit_Score = credit_score
	# Debt_To_Income_Ratio = dti
	# Employment_Length = emp_length
	Loan_Amount = request.args.get('loan_amount')
	Loan_Purpose = request.args.get('loan_purpose')
	Credit_Score = request.args.get('credit_score')
	Debt_To_Income_Ratio = request.args.get('dti')
	Employment_Length = request.args.get('emp_length')

	# preprocessing
	observation = np.array([Loan_Amount, Loan_Purpose, Credit_Score, Debt_To_Income_Ratio, Employment_Length])

	observation[1] = list(categorical_names[1]).index(observation[1])
	observation[4] = list(categorical_names[4]).index(observation[4])
	observation = observation.astype(float)

	# exp = explainer.explain_instance(observation, model.predict_proba(encoder.transform(observation)).astype(float), num_features=8).aslist()
	exp = explainer.explain_instance(observation, predict_fn, num_features=8).as_list()

	output = {
		'Reason_code' : exp
	}
	print(output)
	return json.dumps(output)


@app.route("/")
def hello():
    return "Hello World!"

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=80)
