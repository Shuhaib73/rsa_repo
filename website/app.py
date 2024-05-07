# Import Necessary Libraries
import pandas as pd
import numpy as np 
import re 
import json 
import requests
import pickle
from numpy import asarray, zeros
import string
import signal
import atexit
import os

import cv2 as cv 

# Import Machine Learning libraries
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Import TensorFlow for deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout, Bidirectional, Embedding, SpatialDropout1D, Flatten, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.utils import to_categorical
from keras.initializers import Orthogonal
from tensorflow.keras.models import load_model

# from tensorflow.keras.utils import plot_model
# from tensorflow.python.keras.utils.np_utils import to_categorical

# Import NLTK Libraries for natural language processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import FreqDist
from nltk.stem import PorterStemmer, WordNetLemmatizer

from model2 import Predict_Message

# Model 3 
from COM.image_emotion import EmotionDetector


from flask import Flask, render_template, request, flash, redirect, url_for, get_flashed_messages, jsonify, session, Response

from flask_cors import CORS
import subprocess

# Model 1 ***********************

stopwords_list = set(stopwords.words('english'))
max_length = 100
embedding_matrix = ''
weights=[embedding_matrix]

# Load the model and custom Function
pretrained_lstm_model = load_model('LSTM_Model.h5')

# Function for data pre-processing
from Preprocessing import CustomPreprocess
custom = CustomPreprocess()

# Function for tokenization
with open('word_tokenizer.json') as f:
    data = json.load(f)
    loaded_tokenizer = tokenizer_from_json(data)

# **********************************

# Model 3 files
Model = 'COM/Emotion_Detection_Model.h5'
FaceModel = 'COM/haarcascade_frontalface_default.xml'
detector = EmotionDetector(Model, FaceModel)

# Create the app object
app = Flask(__name__)

app.config['SECRET_KEY'] = 'abcdef'

# ****************************************
# Define Rasa server command
RASA_SERVER_COMMAND = "rasa run --enable-api --model models/20240502-093602-atomic-rate.tar.gz --cors '0.0.0.0:*' --debug --port 5005"

# RASA_SERVER_COMMAND = "rasa run -m models --enable-api --cors "*" --debug -p 5005"

RASA_API_URL = "http://localhost:5005/webhooks/rest/webhook"

# Rasa action server
RASA_ACTION_SERVER = "rasa run actions"


# # Start Rasa server when app starts
rasa_process = None

def start_rasa_server():
    global rasa_process
    if rasa_process is None:
        rasa_process = subprocess.Popen(RASA_SERVER_COMMAND, shell=True)
        rasa_action_server_process = subprocess.Popen(RASA_ACTION_SERVER, shell=True)


# ****************************************


@app.route('/')
def index():
    return redirect('/home')


@app.route('/home')
def home():
    return render_template('base.html')


@app.route('/Sentiment Page', methods=['GET', 'POST'])
def Model1():

    if request.method == 'POST':
        # Get all the values submitted via the form and convert them to strings
        query_asis = [str(x) for x in request.form.values()]

        # Preprocess review text with preprocess_text function
        query_processed_list = []
        for query in query_asis:
            query_processed = custom.preprocess_text(query)
            query_processed_list.append(query_processed)
    
        # Tokenising instance with trained tokenizer
        query_tokenized = loaded_tokenizer.texts_to_sequences(query_processed_list)

        # Padding instance to have maxlength of 100 tokens
        query_padded = pad_sequences(query_tokenized, padding='post', maxlen=max_length)

        # passing tokenized instance to the LSTM Model for prediction
        query_sentiment = pretrained_lstm_model.predict(query_padded)

        if query_sentiment[0][0] > 0.5:
            msg = f"😍 Positive Review with the Predicted IMDB Rating : {np.round(query_sentiment[0][0] * 10, 1)}"
            flash(msg, category='success')
        else:
            msg = f"😡 Negative Review with the Predicted IMDB Rating : {np.round(query_sentiment[0][0] * 10, 1)}"
            flash(msg, category='error')


    return render_template('Model1_sentiment.html')


@app.route('/Detection Page', methods=['GET', 'POST'])
def Model2():
    if request.method == 'POST':
        Mail_Message = request.form['mail_msg']

        if len(Mail_Message) > 0:

            pred = Predict_Message('Spam_Detection_Pipeline')

            prediction = pred.predict(Mail_Message)

            if prediction == 0:
                msg = f"📩✅ This is a Ham Message"
                flash(msg, category='success')
            else:
                msg = f"📩🚫 This is a Spam Message"
                flash(msg, category='error')

    return render_template('Model2_SpHm.html')


@app.route('/Emotion Detection', methods=['POST', 'GET'])
def Model3():

    if request.method == 'POST':
        if 'image' in request.files:
            try:
                image = request.files['image']
                image_path = "static/images/" + image.filename
                image.save(image_path)

                # Detect emotions using EmotionDetector class
                annotated_image = detector.detect_emotions(image_path)

                annotated_image_path = "static/images/annoted_images/" + image.filename
                cv.imwrite(annotated_image_path, annotated_image)

                # Return the path to the annotated image file
                return render_template("image_result.html", annotated_image_path=annotated_image_path)
            except Exception as e:
                print("Error:", e)  # Print the error for debugging
                return render_template('Model3.html', error_message="Please upload an image")
        
        elif 'submit_button' in request.form:
            
            labels_dic = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
            face_cascade = cv.CascadeClassifier(FaceModel)
            model = load_model(Model)

            video = cv.VideoCapture(0)

            while True:
                ret, frame = video.read()
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

                for x, y, w, h in faces:
                    sub_face_img = gray[y:y+h, x:x+w]
                    
                    resized = cv.resize(sub_face_img, (48, 48))
                    normalize = resized / 255.0
                    reshaped = np.reshape(normalize, (1, 48, 48, 1))  #len(num_of_img), img_h, img_w, img_color
                    
                    result = model.predict(reshaped)

                    label=np.argmax(result, axis=1)[0]
                    print(label)

                    cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
                    cv.rectangle(frame, (x, y), (x+w, y+h), (50, 255, 50), 2)
                    cv.rectangle(frame, (x, y-40), (x+w, y), (50, 255, 50), -1)

                    cv.putText(frame, labels_dic[label], (x, y-10), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

                cv.imshow("Emotion Detection", frame)
                k = cv.waitKey(1)

                if k==ord('q'):
                    break
            video.release()
            cv.destroyAllWindows()
        
    return render_template('Model3.html')
        


@app.route('/webhook', methods=['POST', 'GET'])
def webhook():

    start_rasa_server()

    if request.method == 'POST':
        user_input = request.form['msg']
        print("User Input:", user_input)


        rasa_response = requests.post(RASA_API_URL, json={'message': user_input})

        rasa_response_json = rasa_response.json()

        print("Rasa Response:", rasa_response_json)

        bot_response = rasa_response_json[0]['text'] if rasa_response_json else 'Sorry, I didn\'t understand that.'

        return jsonify({'response': bot_response})

    return render_template('care_rasa.html')


            
    # if request.method == 'POST':
    #     user_input = request.form['msg']
    #     print("User Input", user_input)

    #     response_from_rasa = user_input

    #     return jsonify({'response': response_from_rasa})

    # return render_template('care_rasa.html')

 
@app.route('/Contact')
def Contact():

    return render_template('Contact.html')


@app.route('/Setting')
def Setting():

    return render_template('Setting.html')


@app.route('/Logout')
def Logout():

    # Clear the session
    session.clear()
    return render_template('Logout.html')



if __name__ == "__main__":
    app.run(host='0.0.0.0')
