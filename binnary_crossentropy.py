import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, SpatialDropout1D
from keras.datasets import imdb
import tensorflow as tf

# Text analysis using recurrent neural networks
np.random.seed(42)
# Maximum number of words in the dictionary
max_features = 5000
# Length of a review in words
maximumlen = 80

c = []
for d in ['/device:GPU:0', '/device:GPU:1']:
    with tf.device(d):
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
        X_train = sequence.pad_sequences(X_train, maxlen=maximumlen)
        X_test = sequence.pad_sequences(X_test, maxlen=maximumlen)

        model = Sequential()

        model.add(Embedding(max_features, 32))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

        # layer for classification
        model.add(Dense(1, activation="sigmoid"))
        # Adam — adaptive moment estimation - optimization algorithm
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(X_train, y_train, batch_size=64, epochs=9, validation_data=(X_test, y_test), verbose=2)
        scores = model.evaluate(X_test, y_test,
                                batch_size=64)
        print("The accuracy of the model on test data : %.2f%%" % (scores[1] * 100))
with tf.device('/cpu:0'):
    sum = tf.device(c)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(sum))
