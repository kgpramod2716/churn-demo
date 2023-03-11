import pickle
from flask import Flask , request,app,jsonify,url_for,render_template
import pandas as pd
import numpy as np

app=Flask(__name__)
model5=pickle.load(open('model.pkl','rb'))
scaler=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['post'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))   
    output=model5.predict(newdata) 
    print(output[0]) 
    return jsonify(output[0])

if__name__=="__main__":
app.run(debug=True)
    
          