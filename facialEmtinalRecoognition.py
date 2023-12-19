# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 22:43:48 2023

@author: DJO
"""
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


dataset_path = r'C:\Users\DJO\Desktop\python project\train'

emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral','sad', 'surprised']

data = []
labels = []

for emotion_label, emotion in enumerate(emotions):
    emotion_folder = os.path.join(dataset_path, emotion)
    for filename in os.listdir(emotion_folder):
        img_path = os.path.join(emotion_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        data.append(img)
        labels.append(emotion_label)

data = np.array(data)
labels = np.array(labels)


data = data / 255.0
data = data.reshape(-1, 48, 48, 1)

label_encoder = LabelEncoder()
labels = to_categorical(label_encoder.fit_transform(labels))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_true_labels, y_pred_labels)
print(f"Accuracy on test set: {accuracy}")

# Sauvegarder le modèle
model.save('model.hdf5')
