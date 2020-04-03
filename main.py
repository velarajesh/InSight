import flask
from flask import Flask
from flask_cors import CORS
from tensorflow import keras
import cv2
import numpy as np
import os.path
from face_detector import Face_Cropper
import json
# ALL CREDIT FOR FER MODEL GOES TO: ivadym on github
# https://github.com/ivadym/FER

emotions = ['Anger', 'Disgust', 'Fear',
            'Happiness', 'Sadness', 'Surprise', 'Neutral']
model = keras.models.load_model("model/ResNet-50.h5", compile=False)
google_vision = Face_Cropper()
UPLOAD_FOLDER = "/home/mentalhealthapi/keras-fer-gcloud/tmp"
DATA_FOLDER ="/home/mentalhealthapi/keras-fer-gcloud/data"


app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER

def preprocess_input(image):
    # Resizing images for the trained model
    image = cv2.resize(image, (197, 197))
    ret = np.empty((197, 197, 3))
    ret[:, :, 0] = image
    ret[:, :, 1] = image
    ret[:, :, 2] = image
    x = np.expand_dims(ret, axis=0)  # (1, XXX, XXX, 3)
    x -= 128.8006  # np.mean(train_dataset)
    x /= 64.6497  # np.std(train_dataset)
    return x

def json_dbstore(data, rw, filename="datastore.json"):
    with open(os.path.join(
                app.config['DATA_FOLDER'], filename), rw) as json_file:
                if rw == 'r':
                    return json.load(json_file)
                elif rw == 'w':
                    json.dump(data, json_file)
    return True
    
@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    global google_vision
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image_file = flask.request.files["image"]
            image_file.save(os.path.join(
                app.config['UPLOAD_FOLDER'], "img.jpg"))

        
            crop_img, goog_data = google_vision.get_cropped_face(
                "tmp/img.jpg", 4)
            open_cv_image = np.array(crop_img)
            # Convert RGB to BGR
            open_cv_image = open_cv_image[:, :, ::-1].copy()

            # npimg = np.fromstring(image, np.uint8)
            # convert numpy array to image
            # open_cv_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            processed_input = preprocess_input(open_cv_image)
            global model
            global emotions
            prediction = model.predict(processed_input)
            prediction_mapped = dict()
            for i in range(len(emotions)):
                value = float(prediction[0][i])
                if value > 1.5:
                    value = 1.5
                prediction_mapped[emotions[i]] = value
            #{emotions[i]: float(
                #prediction[0][i]) for i in range(len(emotions))}
            
            db_data = json_dbstore(None, 'r')
            db_data_tmp = dict()
            # print(db_data)
            for k, v in db_data.items():
                # print(type(db_data))
    
                # print(type(goog_data))
                db_data_tmp[k] = v + prediction_mapped[k] + (goog_data[k] if k in goog_data else 0)
            json_dbstore(db_data_tmp, 'w')
            
            app_log = json_dbstore(None, 'r', filename="appdbstore.json")
            app_name = flask.request.form['app']
            if not app_name in app_log:
                app_log[app_name] = {'cap_count':0, 'emotion':[]}
            app_log[app_name]['cap_count'] += 1
            app_log[app_name]['emotion'].append(max(db_data_tmp, key=db_data_tmp.get))
            
            json_dbstore(app_log, 'w', filename="appdbstore.json")

            data["success"] = True
    return flask.jsonify(data)


@app.route('/data')
def get_data():
    all_data = {"compiled_emotions":json_dbstore(None, 'r'), "app_usage":json_dbstore(None, 'r', filename="appdbstore.json")}

    return flask.jsonify(all_data)

@app.route('/zero')
def zero_data():
    data= {"success": False}
    db_data = json_dbstore(None, 'r')
    db_data_z = {k: 0 for k,v in db_data.items()}
    json_dbstore(db_data_z, 'w')
    
    app_data = json_dbstore(None, 'r', filename='appdbstore.json')
    app_data_z = {"app_name":app_data["app_name"]}
    #app_data_z = {app_usage:{app_name: {cap_count:0, emotion:[]}}}
    json_dbstore(app_data_z, 'w', filename='appdbstore.json')

    data["success"] = True
    return flask.jsonify(data)

@app.route('/')
def greeter():
    return 'mental_health_api'


if __name__ == '__main__':
    # load_model()
    app.run(host='0.0.0.0', port=5000)
