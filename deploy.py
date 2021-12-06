from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
from matplotlib.image import imread
import cv2

# Keras
from keras.models import model_from_json

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

json_file = open("models/third_model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("models/third_model.h5")

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Model loaded. Check http://127.0.0.1:5000/')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        file_path = os.path.join('./uploads', secure_filename(f.filename))
        f.save(file_path)
        
        #img processing
        img=imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),2)
        th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cv2.imwrite(file_path , res)
        

        # Make prediction
        preds = model_predict(file_path, model)
        alp=['Blank','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        print(str(preds))
        y = np.where(preds[0]==1.)
        return alp[y[0][0]]
    return None

def model_predict(img_path, model):
    img = imread(img_path)
    x=cv2.resize(img,(310,310))
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=-1)

    preds = model.predict(x)
    return preds


if __name__ == '__main__':
    app.run(debug=False)