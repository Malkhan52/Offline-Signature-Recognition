import os,io
from flask import Flask, render_template, request, jsonify

from IPython.display import display
from PIL import Image
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

userName = 'User'
userId = 0

__author__ = 'ibininja'

app = Flask(__name__)
model = None
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# def get_model():
#     global model
#     model = load_model('one-class-sign-CNN_one1.h5')
#     print(" Model loaded.")

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload_genuine", methods=['POST'])
def upload_genuine():
    target = os.path.join(APP_ROOT, 'images/')
    
    global userName
    global userId
    if request.method == 'POST':
        result = request.form
        userName = result['name']
        userId = result['id']
    
    if not os.path.isdir(target):
        os.mkdir(target)
    target2 = os.path.join(target, userName+userId+'/')
    
    if not os.path.isdir(target2):
        os.mkdir(target2)

    target3 = os.path.join(target2, 'genuine/')
    
    if not os.path.isdir(target3):
        os.mkdir(target3)

    for file in request.files.getlist("genuine-file"):
        print(file)
        filename = file.filename
        destination = "/".join([target3, filename])
        print(destination)
        file.save(destination)

    return render_template("complete.html",result = result)

@app.route("/upload_forged", methods=['POST'])
def upload_forged():
    target = os.path.join(APP_ROOT, 'images/')
    
    result = { 'name' : userName, 'id' : userId }

    if not os.path.isdir(target):
        os.mkdir(target)

    if not os.path.isdir(target):
        os.mkdir(target)
    target2 = os.path.join(target, userName+userId+'/')
    
    if not os.path.isdir(target2):
        os.mkdir(target2)

    target3 = os.path.join(target2, 'forged/')
    
    if not os.path.isdir(target3):
        os.mkdir(target3)

    for file in request.files.getlist("forged-file"):
        
        filename = file.filename
        destination = "/".join([target3, filename])
        
        file.save(destination)
    global model
    k.clear_session()
    model = load_model("offline-sign-CNN-01.h5")
    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory('images/'+userName+userId, target_size = (150, 220), batch_size = 32, class_mode = 'binary')

    model.fit_generator(training_set, steps_per_epoch = 10, epochs = 10)

    model.save('offline-sign-CNN-01.h5')
    del model

    return render_template("test.html", result = result)

@app.route("/final", methods=['POST'])
def final():

    target = os.path.join(APP_ROOT, 'images/')
    
    # global userName
    # global userId
    # if request.method == 'POST':
    #     result = request.form
    #     userName = result['name']
    #     userId = result['id']
    
    if not os.path.isdir(target):
        os.mkdir(target)

    if not os.path.isdir(target):
        os.mkdir(target)
    target2 = os.path.join(target, userName+userId+'/')
    
    if not os.path.isdir(target2):
        os.mkdir(target2)

    target3 = os.path.join(target2, 'questioned/')
    
    if not os.path.isdir(target3):
        os.mkdir(target3)

    for file in request.files.getlist("question"):
        
        filename = file.filename
        destination = "/".join([target3, filename])
        
        file.save(destination)

    k.clear_session()
    model = load_model('offline-sign-CNN-01.h5')

    test_image = image.load_img(destination, target_size = (150, 220))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    # training_set.class_indices
    print(result[0][0])
    if result[0][0] >= 0.5:
        prediction = 'Genuine'
    else:
        prediction = 'forged'
    print(prediction)
    return render_template("result.html", result = result[0][0])

if __name__ == "__main__":
    # load_model()
    app.run(port=4555, debug=True)