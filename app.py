from __future__ import division, print_function
# coding=utf-8
import sys
import os
import os
from PIL import Image
import glob
import re
import numpy as np
import cv2
# Keras
from tensorflow.keras.utils import to_categorical
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.layers import  MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.models import Sequential, load_model, Model
from keras.callbacks import ModelCheckpoint
import keras
from Attention import Attention #importing attention layer
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, InputLayer, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
#from classification_models.resnet import ResNet18 #loading propose resnet18 as backbone model
from tensorflow.keras.applications import VGG16, ResNet50 #loading existing VGG16 model
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import sqlite3
import pandas as pd
import numpy as np
import pickle
import sqlite3
import random

import smtplib 
from email.message import EmailMessage
from datetime import datetime



app = Flask(__name__)

#load and display dataset labels
path = "Dataset"
labels = []

for folder in os.listdir(path):
    labels.append(folder)

print("Labels found in dataset:", labels)
#function to get label class id
def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

#load and process dataset image features
if os.path.exists('model/X.txt.npy'):#if images already process then load it
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
else:
    X = []
    Y = []
    #read and process dataset images
    for root, dirs, directory in os.walk(path):#loop all images in dataset
        for j in range(len(directory)):
            name = os.path.basename(root)
            if 'Thumbs.db' not in directory[j]:
                img = cv2.imread(root+"/"+directory[j])#read each image
                img = cv2.resize(img, (128,128))#resize image
                X.append(img)#add image features to X array
                label = getLabel(name)#get label from give image class
                Y.append(label)#add label to Y
    X = np.asarray(X)
    Y = np.asarray(Y)
    np.save('model/X.txt',X)
    np.save('model/Y.txt',Y)            
print("Dataset Tongue Images Loading Completed")
print("Total images found in dataset : "+str(X.shape[0]))

#dataset preprocessing such as shuffling and normalization
X = X.astype('float32')
X = X/255 #normalizing images
indices = np.arange(X.shape[0])
np.random.shuffle(indices)#shuffling images
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)
print("Dataset Normalization & Shuffling Process completed")

#now splitting dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
print()
print("Dataset train & test split as 80% dataset for training and 20% for testing")
print("Training Size (80%): "+str(X_train.shape[0])) #print training and test size
print("Testing Size (20%): "+str(X_test.shape[0]))
print()

# Check if pre-trained models exist, if yes load them
if os.path.exists('model/resnet_model.h5') and os.path.exists('model/hybrid_model.h5') and os.path.exists('model/random_forest_model.pkl'):
    print("Loading pre-trained models...")
    resnet_model = load_model('model/resnet_model.h5', custom_objects={'Attention': Attention})
    hybrid_model = load_model('model/hybrid_model.h5', custom_objects={'Attention': Attention})
    print("Hybrid model output shape:", hybrid_model.output_shape)  # ✅ ADD HERE
    rf = pickle.load(open('model/random_forest_model.pkl', 'rb'))
    print("Models loaded successfully!")
else:
    print("Training models from scratch...")
    resnet_model = ResNet50(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), weights='imagenet', include_top=False)
    #for layer in resnet_model.layers:
        #layer.trainable = False
    #add teacher and student layers to backbone resnet18 model
    resnet_model = Sequential() #create CNN model
    #define layer which will act as Teacher model on noisy and clean features from X input data
    resnet_model.add(InputLayer(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    #teacher model will use Conv2D layer to filter out all noisy and clean features
    resnet_model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
    #teacher module will use max layer to collect all noisy and clean features and then assigned high threshold 
    resnet_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    #another Conv2d layer for further filtration
    resnet_model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
    resnet_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    resnet_model.add(BatchNormalization())
    #studnet CNN module to get optimized features from teacher module and then use prediction or output layer 
    #to calculate prediction thrshold and then select predicted label with high probability
    resnet_model.add(Conv2D(70, (1, 1), activation='relu', strides=(1, 1), padding='same'))
    resnet_model.add(MaxPool2D(pool_size=(1, 1), padding='valid'))
    resnet_model.add(BatchNormalization())
    resnet_model.add(Attention(return_sequences=False, name='attention'))
    #flatten layer to convert 4D output to 1D
    resnet_model.add(Flatten())
    #output layer
    resnet_model.add(Dense(units=100, activation='relu'))
    resnet_model.add(Dense(units=100, activation='relu'))
    resnet_model.add(Dropout(0.25))
    resnet_model.add(Dense(units=y_train.shape[1], activation='softmax'))
    #train and compile the model
    resnet_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    hist = resnet_model.fit(X_train, y_train, batch_size = 2, epochs = 20, validation_data=(X_test, y_test), verbose=1)
    #if os.path.exists("model/resnet_weights.hdf5") == False:
        #model_check_point = ModelCheckpoint(filepath='model/resnet_weights.hdf5', verbose = 1, save_best_only = True)
    hybrid_model = Model(
    inputs=resnet_model.input,
    outputs=resnet_model.get_layer('flatten').output
)
    print("Hybrid model output shape:", hybrid_model.output_shape)  # ✅ ADD HERE
    hybrid_model.summary() 
    hybrid_features = hybrid_model.predict(X)  #extracting resnet features
    Y1 = np.argmax(Y, axis=1)
    #split features into train and test
    X_train, X_test, y_train, y_test = train_test_split(hybrid_features, Y1, test_size=0.2) #split dataset into train and test
    #now create Random Forest object
    rf = RandomForestClassifier()
    #trained on Resnet18 features
    rf.fit(X_train, y_train)
    
    #save models for later use
    resnet_model.save('model/resnet_model.h5')
    hybrid_model.save('model/hybrid_model.h5')
    pickle.dump(rf, open('model/random_forest_model.pkl', 'wb'))


UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# function to check the file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

from tensorflow.keras.models import Model, load_model


#fea = load_model('model.h5')


    
   
@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')


@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')



@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/predict2', methods=['GET','POST'])
def predict2():
    try:

        if request.method == "POST":

            print("Entered")

            file = request.files['file']

            if file.filename == '':
                return "No file selected"

            filename = secure_filename(file.filename)
            print("File name:", filename)

            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Read image
            image = cv2.imread(file_path)

            if image is None:
                return "Image not loaded"

            # Resize like training
            image = cv2.resize(image, (32,32))

            img = np.array(image)
            img = img.astype("float32") / 255.0

            # Add batch dimension
            img = np.expand_dims(img, axis=0)

            print("Input shape:", img.shape)

            # Feature extraction
            features = hybrid_model.predict(img)
            print("Features shape:", features.shape)

            # Random Forest prediction
            predict = rf.predict(features)
            predict = predict[0]

            colour = labels[predict]

            disease_map = {
                "DarkRed": "Heart or Kidney Disorder",
                "LightRed": "Healthy / Normal",
                "Purple": "Blood Circulation Problem",
                "Red": "Body Heat or Infection"
            }

            disease = disease_map.get(colour, "Unknown")

            result_img = cv2.imread(file_path)
            result_img = cv2.resize(result_img, (600,400))

            cv2.putText(result_img, 'Colour : ' + colour, (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

            cv2.putText(result_img, 'Disease : ' + disease, (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            output_path = "static/result.jpg"
            cv2.imwrite(output_path, result_img)

            return redirect(output_path)

    except Exception as e:
        print("ERROR:", str(e))
        return str(e)

    return render_template('index.html')
@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "evotingotp4@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("evotingotp4@gmail.com", "xowpojqyiygprhgr")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict1', methods=['POST'])
def predict1():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signin.html")

@app.route("/notebook")
def notebook1():
    return render_template("TongueColourPrediction.html")


   
if __name__ == '__main__':
    app.run(debug=True)
         

