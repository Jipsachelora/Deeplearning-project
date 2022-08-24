#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model
import seaborn as sns
from matplotlib import pyplot as plt
import os
from sklearn.metrics import confusion_matrix


import shutil
#from PIL import Image   
#import tensorflow as tf
#import h5py
import numpy as np



def training_cnn():
    classifier = Sequential()

    # Adding convolution using Conv2D
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

    # Pooling using MaxPooling2D  to a size of 2,2
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # adding Flattening
    classifier.add(Flatten())

    # establishing full connection each layer activated with relu and sigmoid respectievely
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))

    # Compiling CNN using adam optimizer and binary_crossentropy is added for loss
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting the model to imagesspllitting train and test



    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size = (64, 64),
                                                     batch_size = 32,
                                                     class_mode = 'binary')

    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

    classifier.fit_generator(training_set,
                             steps_per_epoch = 1910,
                             epochs = 25,
                             validation_data = test_set,
                             validation_steps = 538)
    #classifier.save("cnnNatureOrCities")

def cnn_test_image(pathnam):
    test_image = image.load_img(pathnam, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)         
    classifier= load_model("./cnnNatureOrCities.hdf5")
    result = classifier.predict(test_image/255)
    #training_set.class_indices
    if result[0][0] == 1:
        prediction = 'Nature'
        #Images categorised as Nature can be found in the outputs/nature folder
        shutil.copy(pathnam, "outputs/nature")
        return prediction
    else:
        prediction = 'Cities'
        #Images categorised as Cities can be found in the outputs/cities folder
        shutil.copy(pathnam, "outputs/cities")
        return prediction
def make_confusion_matrix(cf,
                          title='CONFUSION MATRIX',
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues'):


    # Writing text in each square
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # Summary Stats
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # Fig size definition
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # Seaborn heat visualisation
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
   
    if title:
        plt.title(title)
def cnn_analyse():
    y_true = []
    y_pred = []
    for root, dirs, files in os.walk(os.path.abspath("dataset/test_set/Cities")):
        for file in files:
            print(os.path.join(root, file))
            y_true.append(0)
            if cnn_test_image(os.path.join(root, file))=="Nature":
                y_pred.append(1);
            else:
                y_pred.append(0)
    for root, dirs, files in os.walk(os.path.abspath("dataset/test_set/Nature")):
        for file in files:
            print(os.path.join(root, file))
            y_true.append(1)
            if cnn_test_image(os.path.join(root, file))=="Nature":
                y_pred.append(1);
            else:
                y_pred.append(0)
    print(y_true)
    print(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    labels = ["True Neg","False Pos","False Neg","True Pos"]
    categories = ["Cities", "Nature"]
    make_confusion_matrix(cm,
                      group_names=labels,
                      categories=categories)
   
cnn_analyse()    
