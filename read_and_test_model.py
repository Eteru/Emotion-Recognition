from __future__ import division, print_function, absolute_import

# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

import tensorflow

from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image
import numpy
import pickle
import os, os.path, time


def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def convert_2D_to_3D(array):
    arr = [[0 for x in range(len(array))] for y in range(len(array[0]))]
    #print (arr)
    #print (len(array), len(array[0]))
    #print (len(arr), len(arr[0]))
    for i in range(len(array)):
        for j in range(len(array[0])):
            #print (i, j)
            aux = array[i][j]
            arr[j][i] = [aux, aux, aux]
    return numpy.asarray(arr, dtype='float64')

def initialize_network():

    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Create extra synthetic training data by flipping, rotating and blurring the
    # images on our data set.
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)
    img_aug.add_random_blur(sigma_max=3.)

    # Define our network architecture:

    # Input is a 32x32 image with 3 color channels (red, green and blue)
    network = input_data(shape=[None, 128, 128, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)

    dropout_chance = 0.5

    print ("Step 1")

    # Step 1: Convolution
    network = conv_2d(network, 32, 3, activation='relu')

    print ("Step 2")

    # Step 2: Max pooling
    network = max_pool_2d(network, 2)

    #network = dropout(network, dropout_chance)

    print ("Step 3")

    # Step 3: Convolution again
    network = conv_2d(network, 64, 3, activation='relu')
    #network = max_pool_2d(network, 2)
    #network = dropout(network, dropout_chance)

    print ("Step 4")

    # Step 4: Convolution yet again
    network = conv_2d(network, 64, 4, activation='relu')
    #network = max_pool_2d(network, 2)
    #network = dropout(network, dropout_chance)
    print ("Step 5")

    # Step 5: Max pooling again
    network = max_pool_2d(network, 2)

    print ("Step 6")

    # Step 6: Fully-connected 512 node neural network
    network = fully_connected(network, 512, activation='relu')

    print ("Step 7")

    # Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
    #network = dropout(network, dropout_chance)

    # Step 6: Fully-connected 512 node neural network
    #network = fully_connected(network, 512, activation='relu')

    print ("Step 7")

    # Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
    network = dropout(network, dropout_chance)

    print ("Step 8")

    # Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
    network = fully_connected(network, 7, activation='softmax')

    print ("Step 9")

    # Tell tflearn how we want to train the network
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    # Wrap the network in a model object
    model = tflearn.DNN(network, tensorboard_verbose=0)

    return model


def load_model(model):

    model.load("model5/model5.tfl")

    return model


def read_image(filename):

    im = Image.open(filename, "r")
    im = im.convert("L")
    im = im.resize((128, 128), Image.ANTIALIAS)
    #im.show()
    #print (im.size)
    imarray = numpy.asarray(im, dtype='float64')/256.
    #print (imarray)
    #print
    imarray = convert_2D_to_3D(imarray)
    im.show(im)
    print([imarray])
    return [imarray]


def predict(array, model):
    
    preds = model.predict(array)

    final_preds = []

    for pred in preds:
        aux = []
        for p in pred:
            aux.append(truncate(p, 4))
        #print ("aux is ", aux)
        final_preds.append(aux)

    return final_preds




