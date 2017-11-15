import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# NN recognizes handwritten figures
# The algorithm receives an image and needs to recognize the correct digit
begin_time = time.time()
numpy.random.seed(42)

# There are 60 thousand images.  Fail with images and image tags
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
# normalize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# marks to category (conversion of correct answers by Category)
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# create model
model = Sequential()

# add hidden layers [3]
model.add(Dense(800, input_dim= 784 , activation="relu", kernel_initializer="normal"))
model.add(Dense(600, activation="relu", kernel_initializer="normal"))
model.add(Dense(10, activation="softmax", kernel_initializer="normal"))

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
print(model.summary())

# batch_size- Size of the mini-selection;
model.fit(X_train, Y_train, batch_size=25, epochs=125, validation_split=0.2, verbose=2)

# evaluate the quality of the network training on the test data
scores = model.evaluate(X_test, Y_test, verbose=0)
print("The accuracy of the model on test data %.2f%%" % (scores[1] * 100))

model_json = model.to_json()
json_file = open("mnist_model.json", "w")
json_file.write(model_json)
json_file.close()
