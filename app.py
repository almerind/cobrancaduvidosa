import pickle
from django.shortcuts import render
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import json

app=Flask(__name__)
## Load the model

with open('regmodelOHE.pkl', 'rb') as f:
    regmodel = pickle.load(f)

with open('scalerOHEF.pkl', 'rb') as k:
    scalar = pickle.load(k)

# regmodel = pickle.load(open('regmodelOHE.pkl'))
# scalar = pickle.load(open('scalarOHE.pkl'))

@app.route('/')
def home():
    return render_template('home3.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data= request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return json.dumps(output[0], default=str)
    # return jsonify(output[0])

@app.route('/predict', methods=['POST']) 
# def predict():
#     data = request.form
#     # final_input=scalar.transform(np.array(data).reshape(1,-1))
#     # print(final_input)
#     # output = regmodel.predict(final_input)[0]
#     return render_template("home1.html",prediction_text="Loan System!! {} ".format(data))

def predict():
    data = [float(x) for x in request.form.values()]
    # final_input=scalar.transform(np.array(data).reshape(1,-1))
    # print(final_input)
    # output = regmodel.predict(final_input)[0]
    return render_template("home3.html",prediction_text="Loan System!! {} ".format(data))
    
if __name__=="__main__":
    app.run(debug=True)