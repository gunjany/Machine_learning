# Creating a front-end web app using flask

from flask import url_for, request, Flask, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import numpy as np
import pickle

app_insurance = Flask(__name__, template_folder = 'template')

model = pickle.load(open('deployment_30052020.pkl', 'rb'))
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@app_insurance.route('/')
def home():
    return render_template("home.html")


@app_insurance.route('/predict', methods = ['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = model.predict(data = data_unseen)
    prediction = int(prediction.Label[0])
    return render_template('home.html', pred = 'Expected Bill will be {}'.format(prediction))

# @app_insurance.route('/predict_api',methods=['POST'])
# def predict_api():
#     data = request.get_json(force=True)
#     data_unseen = pd.DataFrame([data])
#     prediction = predict_model(model, data=data_unseen)
#     output = prediction.Label[0]
#     return jsonify(output)

if __name__ == '__main__':
    app_insurance.run(debug = True)