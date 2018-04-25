# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by downloading the model file [here](https://www.dropbox.com/s/1dsrp5tqrwfftj6/model.h5?dl=0) and then  executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 16 and 32 (model.py lines 8-37)

The model includes ELU layers to introduce nonlinearity.

#### 2. Attempts to reduce overfitting in the model

To avoid overfitting in the model I added dropout between after every convolutionary layer and also I used augmentation by mirroring 50% of the images to mimic driving around the track in opposite direction. All of the images has some brightness augmentation applied to.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

The data I used was the sample data provided by Udacity. I also collected my own data but quickly realized that data only made the model's performance worse. This is possibly due to bad driving or my steering style can be very noisy.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I first tried using the Nvidia model (As can be seen below) and also the Udacity data. I quickly realized I needed to collect more data. The data I collected was not good and made the network perform worse. then I decided to reduce the complexity of the Nvidia model and therefore not needing so much data. I iterated the model structure a couple of times and came up with the one described in section 2 below.

To prevent over fitting I added dropout layers and also max pooling layer.

I splitted the data into 80% training data and 20% validation data. The left and right images was added/decreased an angle of 0.25.

![alt text](./examples/cnn-architecture-624x890.png)

#### 2. Final Model Architecture

The final model architecture can be seen in the table below.

| Layer | Output Shape | Param # | 
|-------|------------|---|---|
|convolution2d_1 (Convolution2D)|(None, 32, 32, 32)|2432|
|elu_1 (ELU)|(None, 32, 32, 32)|0|
|convolution2d_2 (Convolution2D)|(None, 30, 30, 15)|4624|
|elu_2 (ELU)|(None, 30, 30, 16)|0|
|dropout_1 (Dropout) |(None, 30, 30, 16)|0|
|max_pooling2d_1 (MaxPooling2 |(None, 15, 15, 16)|0|
|convolution2d_3 (Convolution2D)  |(None, 13, 13, 16)| 2320 |
|elu_3 (ELU)|(None, 13, 13, 16)|0|
|dropout_2 (Dropout) |(None, 13, 13, 16)|0|
|flatten_1 (Flatten)|(None, 2704) |0 |
|dense_1 (Dense) | (None, 1024)| 2769920|
|dropout_3 (Dropout) |(None, 1024)|
|elu_4 (ELU)|(None, 1024)|0|
|dense_2 (Dense)| (None, 512)|524800 |
|elu_5 (ELU)|(None, 512)|0|
|dense_3 (Dense)|(None, 1)|513|

#### 3. Creation of the Training Set & Training Process

I used all three cameras provided my the data that came from the simulator. Below an image can be seen taken from the center camera.

![alt text](./examples/original.jpg)


To augment the data sat, I also flipped images and angles thinking that this would help to model not to overfit on the data.

![alt text](./examples/mirror.jpg)

All of the images used was exposed to random brightness augmentation.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 

The ideal number of epochs was 3 as evidenced by the loss did not decrease after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.
