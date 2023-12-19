# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 07:56:08 2023

@author: DJO
"""

import cv2
from keras.models import load_model
import numpy as np

emotion_model = load_model(r'C:\Users\DJO\Desktop\python project\model.hdf5')


emotion_labels = {
    0: 'Angry',
    1: 'Disgusted',
    2: 'Afraid',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprised'
}


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        
        roi_gray = gray[y:y + h, x:x + w]

        
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi_gray = roi_gray / 255.0

       
        roi_gray = np.reshape(roi_gray, (1, 64, 64, 1))


        emotion_prediction = emotion_model.predict(roi_gray)
        emotion_index = np.argmax(emotion_prediction)
        emotion_label = emotion_labels[emotion_index]

        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
