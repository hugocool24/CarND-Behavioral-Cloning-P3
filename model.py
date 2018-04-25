import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D, Activation
from keras.preprocessing.image import img_to_array, load_img
import cv2

def keras_model():
    model = Sequential()

    model.add(Convolution2D(32, 5, 5, input_shape=(64, 64, 3), subsample=(2, 2), border_mode="same"))
    model.add(ELU())

    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))

    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Dropout(0.3))
    model.add(ELU())

    model.add(Dense(512))
    model.add(ELU())

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    model.summary()

    return model

def randomImage(row, applied_angle = 0.25):
    angle = row['steering']
    pickCamera = np.random.choice(['center', 'left', 'right'])
    if pickCamera == 'left':
        angle += applied_angle
    elif pickCamera == 'right':
        angle -= applied_angle

    image = load_img("data/" + row[pickCamera].strip())
    image = img_to_array(image)

    #Invert the image to generate different kind of training data
    flipRate = np.random.random()
    if flipRate > 0.5:
        angle = -1*angle
        image = cv2.flip(image, 1)

    #Perform random brightness
    imageBright = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    # Change brightness channel to a random value + 0.25 in offset
    imageBright[:,:,2] = imageBright[:,:,2]*random_bright
    imageBright = cv2.cvtColor(imageBright,cv2.COLOR_HSV2RGB)

    # Crop and resize
    cropImage = imageBright[55:135, :, :]
    cropImage = cv2.resize(cropImage, (64,64))
    cropImage = cropImage.astype(np.float32)
    cropImage = cropImage/255.0 - 0.5

    return cropImage, angle

#Generator to generate batches of images to train/val on
def generator(driveImg, batch_size=64):
    numSamples = driveImg.shape[0]
    batches_per_epoch = numSamples // batch_size
    i = 0

    while True:
        batchStart = i*batch_size
        batchEnd = batchStart+batch_size - 1
        X_train = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
        y_train = np.zeros((batch_size,), dtype=np.float32)
        j = 0

        for counter, row in driveImg.loc[batchStart:batchEnd].iterrows():
            X_train[j], y_train[j] = randomImage(row)
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            i = 0

        yield X_train, y_train

batch_size = 64
applied_angle = 0.25
#Read in CSV file
data_frame = pd.read_csv('data/driving_log.csv', usecols=[0, 1, 2, 3])
#Shuffle the data
data_frame = data_frame.sample(frac=1).reset_index(drop=True)

#Training validation split
num_rows_training = int(data_frame.shape[0]*0.8)
#Train Data
train_images = data_frame.loc[0:num_rows_training-1]
#Val Data
validation_images = data_frame.loc[num_rows_training:]
data_frame = None

trainGen = generator(train_images, batch_size = 64)
validationGen = generator(validation_images, batch_size = 64)

# Train model
model = keras_model()

samples_per_epoch = (20000//batch_size)*batch_size
model.fit_generator(trainGen,
                   samples_per_epoch = samples_per_epoch,
                   validation_data = validationGen,
                   nb_val_samples = 3000,
                   nb_epoch = 3)

model.save("model.h5")
with open("autopilot_game.json", "w") as outfile:
    outfile.write(model.to_json())
