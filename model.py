from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import BatchNormalization,Input, Cropping2D, Dropout, Lambda
#from keras import backend as K
import json
import numpy as np
from skimage import color
from skimage import io
import sklearn
from skimage.transform import rotate, resize
from sklearn.model_selection import train_test_split
import csv
import os
import random
import sys
import tensorflow as tf

def keras_model():

    model = Sequential()
    model.add(Cropping2D(cropping=((50,24), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda images: tf.image.resize_images(images, (66, 200))))
    model.add(Lambda(lambda x: (x/255) - 0.5))
    model.add(Convolution2D(24,5,5,border_mode="valid", activation="relu", subsample=(2,2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(36,5,5,border_mode="valid", activation="relu", subsample=(2,2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(48,5,5,border_mode="valid", activation="relu", subsample=(2,2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64,3,3,border_mode="valid", activation="relu", subsample=(1,1)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64,3,3,border_mode="valid", activation="relu", subsample=(1,1)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1164, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1, activation='tanh'))
    model.summary()
    with open("autopilot_game.json", "w") as outfile:
        outfile.write(json.dumps(json.loads(model.to_json()), indent=2))
    return model

paths =("./Data_curves/","./Data/","./Data_bridge/") #Path to self-collected data
path2=("./data/") #Path to udacity sample data
images = []
applied_angle = 0.25
batch_size = 32

# Make a list of paths to images.
with open(path2 + 'driving_log.csv') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        center = row[0].replace(" ", "")
        left = row[1].replace(" ", "")
        right = row[2].replace(" ", "")
        angle = float(row[3].replace(" ", ""))
        #Pick Center, left, right image randomly
        ### Append non-flipped images
        randomly = random.randint(0,10)
        if randomly<4:
            images.append((path2+center, angle, False))
        elif randomly<6:
            images.append((path2+left, angle + applied_angle, False))
        else:
            images.append((path2+right, angle - applied_angle, False))

        ### Append flipped images randomly
        if randomly<4:
            images.append((path2+center, angle, True))
        elif randomly<6:
            images.append((path2+left, -(angle + applied_angle), True))
        else:
            images.append((path2+right, -(angle - applied_angle), True))

for path in paths:
    with open(path + 'driving_log.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            center = row[0][65:]
            left = row[1][65:]
            right = row[2][65:]
            angle = float(row[3])

            #Randomly pick an image between center, right and left. More center images
            randomly = random.randint(0,10)
            if randomly<4:
                images.append((path+center, angle, False))
            elif randomly<6:
                images.append((path+left, angle + applied_angle, False))
            else:
                images.append((path+right, angle - applied_angle, False))

            #Randomly pick an image between center, right and left. More center images
            ### Append flipped images randomly
            if randomly<4:
                images.append((path+center, angle, True))
            elif randomly<6:
                images.append((path+left, -(angle + applied_angle), True))
            else:
                images.append((path+right, -(angle - applied_angle), True))

#Generator to generate batches of images to train on
def generator(driveImg):
    batch_size = 64
    while True:
        image = sklearn.utils.shuffle(driveImg)
        images = []
        angles = []
        for img in image:
            load_image = io.imread(img[0])
            if img[2] == True:
                load_image = load_image[:, ::-1]
            images.append(load_image)
            angles.append(img[1])
            if  len(images) == batch_size:
                X_train = np.array(images)
                y_train = np.array(angles)
                yield X_train, y_train

#Shuffle data into validation set och training set
train_images, validation_images = train_test_split(images, test_size=0.2)

trainGen = generator(train_images)
validationGen = generator(validation_images)

# Train model
model = keras_model()
model.compile(loss='mse', optimizer='adam')
batch_size = 64
nb_train = len(train_images)
nb_val = len(validation_images)

model.fit_generator(trainGen,
                   samples_per_epoch = nb_train // batch_size,
                   validation_data = validationGen,
                   nb_val_samples = nb_val // batch_size,
                   nb_epoch = 1,
                   verbose = 1)

model.save("model.h5")
