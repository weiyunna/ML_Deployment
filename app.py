import sys
#sys.path.append('C:\\ML_Deployment')
from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

reg_model = "reg_model.sav"
model = pickle.load(open(reg_model, 'rb'))
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = np.array([int_features])    
    prediction = model.predict(final)
    prediction = round(prediction[0], 2)
    #data_unseen = pd.DataFrame([final], columns = cols)
    #prediction = predict_model(model, data=data_unseen, round = 0)
    #prediction = int(prediction.Label[0])
    return render_template('home.html',pred='Expected Bill will be {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    #final = np.array([int_features])    
    prediction = model.predict(data_unseen)
    prediction = round(prediction[0], 2)
    
    #prediction = predict_model(model, data=data_unseen)
    #output = prediction.Label[0]
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)