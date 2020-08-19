#imported libraries
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')

#generating spectrograms
cmap = plt.get_cmap('inferno')
plt.figure(figsize=(8,8))
genres = 'dooropen doorclose airplane carhorn barkingdog'.split()
for g in genres:
    pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(rf"C:\Users\maria\Desktop\Bachelor Thesis\Data\Audio Samples\training data\{g}"):
        songname = rf"C:\Users\maria\Desktop\Bachelor Thesis\Data\Audio Samples\training data\{g}\{filename}"
        y, sr = librosa.load(songname, mono=True, duration=5)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
        plt.clf()
        
#initialising csv file with headers
header = 'filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

#populating csv file, each row containing one sound file (information sourced from the spectrogram)
file = open('dataset.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'dooropen doorclose airplane carhorn barkingdog'.split()
for g in genres:
    for filename in os.listdir(rf"C:\Users\maria\Desktop\Bachelor Thesis\Data\Audio Samples\training data\{g}"):
        songname = rf"C:\Users\maria\Desktop\Bachelor Thesis\Data\Audio Samples\training data\{g}\{filename}"
        y, sr = librosa.load(songname, mono=True, duration=30)
        rms = librosa.feature.rms(y=y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('dataset.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

#preprocessing data
data = pd.read_csv('dataset.csv')
data.head()# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)#Encoding the labels
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))#Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#building the ANN model
model = Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#fitting the model 
epochs = 150
xs, acctest= [], []
classifier = model.fit(X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=128)

#testing the model
for step in range(epochs):
   y_pred = model.predict(X_test)
   xs.append(step)
   acctest.append(model.evaluate(X_test,y_test))
   model.evaluate(X_test,y_test)
   
#saving the model in .hd5 format
model.save(r"C:\Users\maria\Desktop\model.hd5")