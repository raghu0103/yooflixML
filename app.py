#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 01:21:08 2021

@author: Raghwendra Sharan
"""

import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features='Best Scenes of 3 Idiots '
    #prediction = model.make_recommendations(features)
    prediction = model[features].sort_values(ascending=False)[:10]
    #return render_template('index.html', prediction_value='Similiar Videos {}'.format(prediction) )
    return render_template('index.html', prediction_value=prediction )

if __name__== "__main__":
    app.run(debug=True)
    