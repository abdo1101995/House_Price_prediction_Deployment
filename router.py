
from distutils.log import debug
import numpy as np
import pandas as pd
import joblib
from distutils.log import debug
from flask import Flask ,render_template,request
from utils import process_new

model=joblib.load('model_xgb.pkl')
#intialize

app=Flask(__name__)



@app.route('/')

def home():
    return  render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])

def predict():
    if request.method=='POST':
        long=float(request.form['long'])
        lat=float(request.form['lati'])
        house_median=float(request.form['h_med'])
        total_rooms=float(request.form['to_ro'])
        total_bedrooms=float(request.form['to_bro'])
        pop=float(request.form['pop'])
        household=float(request.form['houssh'])
        median_income=float(request.form['med'])
        ocean_proximity=request.form['ocean']
        rooms_per_holds=total_rooms/household
        bedrooms_per_rooms=total_bedrooms/total_rooms
        pop_per_hold=pop/household
        X_new=pd.DataFrame({'longitude':[long],'latitude':[lat],'housing_median_age':[house_median],'total_rooms':[total_rooms],
        'total_bedrooms':[total_bedrooms],'population':[pop],'households':[household],'median_income':[median_income],
        'ocean_proximity':[ocean_proximity],'room_per_houseshold':[rooms_per_holds],'bedromms_per_rooms':[bedrooms_per_rooms],
        'population_per_houseshold':[pop_per_hold]})

        X_processed=process_new(X_new)
        y_pred_new=model.predict(X_processed)
        y_pred_new='{:0.4f}'.format(y_pred_new[0])
        return  render_template('predict.html',pred_vals=y_pred_new)
    else:
        return  render_template('predict.html')

@app.route('/about')

def about():
    return  render_template('about.html')

#terminal
if __name__=='__main__':
    app.run(debug=True)