import pickle
from flask import Flask,render_template,request,jsonify
import numpy as np 
import pandas as pd 

applicaton=Flask(__name__)
app=applicaton

scaler=pickle.load(open('scaler.pkl','rb'))
ridge=pickle.load(open('ridge.pkl','rb'))



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET',"POST"])
def data():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        WS=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region =float(request.form.get('Region'))
        FFMC=float(request.form.get('FFMC'))
        new_data=scaler.transform([[Temperature,RH,WS,Rain,DMC,FFMC,ISI,Classes,Region]])
        result=ridge.predict(new_data)
        
        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')

if __name__=='__main__':
     app.run(debug=True)