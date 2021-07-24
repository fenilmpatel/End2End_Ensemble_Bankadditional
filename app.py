from flask import Flask,render_template,url_for,request
import numpy as np
import pandas as pd
import os
import pickle
model = pickle.load(open('xgboost.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():

    return render_template('index1.html',title='Home')

@app.route('/predict',methods=['POST'])
def predict():
    '''For rendering results on HTML GUI
    '''
    month = request.form['month']
    emp = request.form['emp']
    cons  = request.form['cons']
    contact = request.form['contact']
    housing =request.form['housing']
    euribor3m = request.form['euribor3m']
    default = request.form['default']

    pred = pd.DataFrame(data={'month':[float(month)],'emp.var.rate':[float(emp)] ,'cons.conf.idx':[float(cons)],'contact':[float(contact)],
                       'housing':[float(housing)],'euribor3m':[float(euribor3m)],'default': [float(default)]})
    prediction = model.predict(pred)
    output = prediction[0] 
    if output > 0:
        output="Genuine"
        return render_template('prediction.html', prediction_text=f'Prediction For Applied Person is {output} Person.')
    else:
        output = "Fraud"
        return render_template('prediction.html', prediction_text=f'Prediction For Applied Person is {output} Person!!')
   
port = int(os.environ.get('PORT',5000))
if __name__ == "__main__":
    app.run(debug=1,host='0.0.0.0',port=port) # or True             
