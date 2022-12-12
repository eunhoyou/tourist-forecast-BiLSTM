import tensorflow as tf
import numpy as np
import pickle
import flask
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from keras.models import load_model
from pickle import load
import json

import boto3
from m_config import AWS_S3_BUCKET_REGION, AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY
from m_config import AWS_S3_BUCKET_NAME 

app = flask.Flask(__name__)

def s3_connection():
    try:
        s3 = boto3.client(
            service_name = 's3',
            region_name = AWS_S3_BUCKET_REGION,
            aws_access_key_id = AWS_ACCESS_KEY,
            aws_secret_access_key = AWS_SECRET_ACCESS_KEY
        )
    except Exception as e:
        print(e)
        exit("connect failed")
    else:
        print("s3 bucket connected")
        return s3
    
def s3_get_object(bucket, object_name, file_name):
    s3 = s3_connection()
    
    try:
        s3.download_file(bucket, object_name, file_name)
    except Exception as e:
        print(e)
        return False
    return True

def load_model_from_S3(signguCode):
    key = 'model/model_BiLSTM_' + str(signguCode) + '.h5'
    file_name = 'model_BiLSTM_' + str(signguCode) + '.h5'
    
    s3_get_object(AWS_S3_BUCKET_NAME, key, file_name)
    
    model = load_model('model_BiLSTM_' + str(signguCode) + '.h5')
    return model

def load_scaler_from_S3(signguCode):
    key = 'scaler/scaler_' + str(signguCode) + '.h5'
    file_name = 'scaler_' + str(signguCode) + '.h5'
    
    key2 = 'scaler/scaler2_' + str(signguCode) + '.h5'
    file_name2 = 'scaler2_' + str(signguCode) + '.h5'
    
    s3_get_object(AWS_S3_BUCKET_NAME, key, file_name)
    s3_get_object(AWS_S3_BUCKET_NAME, key2, file_name2)
    
    scaler = load(open('scaler_' + str(signguCode) + '.pkl', 'rb'))
    scaler2 = load(open('scaler2_' + str(signguCode) + '.pkl', 'rb'))
    return scaler, scaler2

@app.route("/predict", methods=["POST"])
def predict():
    req = flask.request.get_json()
    
    month = req['userRequest']['month']
    day = req['userRequest']['day']
    signguCode = req['userRequest']['signguCode']
    daywkDivCd = req['userRequest']['daywkDivCd']
    isHoliday = req['userRequest']['isHoliday']
    isCovid = req['userRequest']['isCovid']
    
    model = load_model_from_S3(signguCode)
    scaler, scaler2 = load_scaler_from_S3(signguCode)
    
    x = np.array([[month, day, signguCode, daywkDivCd, isHoliday, isCovid]])
    x_scaled = scaler.fit_transform(x)
    x = np.expand_dims(x_scaled, axis=0)
    y = model.predict(x)
    result = int(scaler2.inverse_transform(y[0]))
    
    return flask.jsonify({'result' : result}), 200

if __name__ == "__main__":
    app.run()