#
# SciNet's DAT112, Neural Network Programming.
# Assignment 1 :
#   Sort cat and dog images into respective categories using Neural Networks
#   Saved as dogs_cats_nn.py
# By: Valeriya Dontsova
# Date: May 21, 2022
#

#############
"""
dogs_cats_nn.py loads Dogs vs. Cats data set, splits it into training and testing data, and trains a
neural network on these data to distinguish between dogs and cats.
"""
#############

# Perform all the imports of necessary packages
import tensorflow
import numpy
import sklearn.model_selection as skms
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.utils as ku
import matplotlib.pyplot as plt


# Build the NN model
def get_model(numfm, input_shape = (50,50,3), output_size = 2, d_rate = 0.1):
    """
    This function returns a Keras model taking RBG images as input and returns a binary classification.
    Model consists of an input layer, numfm feature maps/convolutional layers, pooling layers, 1 hidden layer and an output level

    Inputs:
    - numfm: int, the number of feature maps connected to the first convolution layer.

    - input_shape: tuple, dimensions of image and colour channels, default - 50x50 pixels, 3 RBG channels (50,50,3)

    - output_size: int, number of nodes in the output layer, default = 2

    - d_rate = dropout rate

    Output: Keras model.

    """
    # initalize a sequential models
    model = km.Sequential()

    # add a convolutional layer
    model.add(kl.Conv2D(numfm, kernel_size = (4, 4), input_shape = input_shape, activation = "relu"))

    # pool the feature maps
    model.add(kl.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    # add dropout of neurons to avoid overfitting
    model.add(kl.Dropout(d_rate*2))

    # repeat above code for 1 more convolutional layer whilst increasing number of feature maps
    model.add(kl.Conv2D(numfm*2, kernel_size = (2, 2), input_shape = input_shape, activation = "relu"))
    model.add(kl.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    # add dropout of neurons
    model.add(kl.Dropout(d_rate*3))

    # additional convolutional layers
    model.add(kl.Conv2D(numfm*4, kernel_size = (2, 2), input_shape = input_shape, activation = "relu"))
    model.add(kl.Conv2D(numfm*6, kernel_size = (2, 2), input_shape = input_shape, activation = "relu"))
    #pool the layers
    model.add(kl.MaxPooling2D(pool_size = (4, 4), strides = (1, 1)))

    # flaten the 3D array
    model.add(kl.Flatten())

    # add batch normalization for the upcoming dense layer
    model.add(kl.BatchNormalization(name = 'batch_normalization'))

    # creation and addition of an output layer, fully connected
    # one dense layer appears to perform better than 2
    model.add(kl.Dense(output_size, activation = 'softmax', name = "output_layer"))

    # return the model
    return model


# load the data
# cats_n_dogs is returned as a dictionary of 50x50 pixel RBG images and corresponding classification labels
data_name = "resized_dogs_cats_50.npz"
print("Reading dogs vs. cats data file.")
cats_n_dogs = numpy.load(data_name)

# split into images and labels
images, labels = cats_n_dogs["images"], cats_n_dogs["labels"]

# split the data into training and testing data based of 80% to 20% split respectively
train_im, test_im, train_lab, test_lab = skms.train_test_split(images, labels, test_size = 0.2)

# one-hot label encoding from single value to 2 point array corresponding to output neurons
train_lab = ku.to_categorical(train_lab, 2)
test_lab = ku.to_categorical(test_lab, 2)

# Build the NN model
print("Building network.")
# value indicates the number of feature maps
model = get_model(30)
model.summary()
#compile model with crossentropy cost function
model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = "categorical_crossentropy")

# train the model and print results
print("Training network.")
fit = model.fit(train_im, train_lab, epochs = 50, batch_size = 500, verbose = 0) # convert to 0 for final submission

train_score = model.evaluate(train_im, train_lab)
print("The training score is", train_score)

# test the model on the testing set and print results
test_score = model.evaluate(test_im, test_lab)
print("The testing score is", test_score)

#plot the data
plot1 = plt.figure() # used to plot the data when training
plt.plot(fit.history['accuracy'])
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.show() # used to plot the data when training
plot1.savefig('plot1.pdf') # used to save the plot when training
