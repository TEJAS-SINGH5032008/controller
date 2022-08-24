import numpy as np
import pandas as pd
import sklearn.datasets import fetch_openml
import sklearn.model_selection import train_test_split
import sklearn.linear_model import LogisticRegression
import pil from image
import pil.imageOPS
from flask import Flask, jsonify, request
from classifier import get_prediction
  
  app = Flask(_name_)
@app.route("/predict-alphabet", methods = ["POST"])
def predict_data():
    image = request.files.get("alphabet")
    prediction = get_prediction(image)
    return jsonify({
        "prediction": prediction
    }), 200
    if _name_ == "_main_":
        app.run(debug = True)
X, y = fetch_openml('mnist_784' , version=1 , return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9 , train_size = 7500 , test_size = 2500)

X_train_scaled = X_train./255.0
X_test_scaled = X_test/255.0

clf = LogisticRegression(solver = 'saga', mullti_class='multinomial').fit(X_train_scaled, y_train)

def get_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized , pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    image_bw_resized_inverted_scaled  = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_pred = clf.predict(test_sample)
    return test_pred[0]