'''Uses a not-recurrent network to do timeseries prediction on feature vectors
I don't know why but this is faster and seems to work better than RNN
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, SimpleRNN
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

np.set_printoptions(suppress=True)

def train_nn(
    corpus,
    memory = 3,
    step = 1,
    learn_rate = 0.01,
    epochs = 2000,
    batch_size = 20,
    sample_length = 10,
    callback_step = 100,
    callback = lambda i, features: print(i, features)
    ):
    # cut the text in semi-redundant sequences of memory characters
    print('corpus length:', len(corpus))
    
    assert memory < len(corpus)

    sentences = []
    next_values = []
    for i in range(0, len(corpus) - memory, step):
        sentences.append(corpus[i: i + memory].flatten())
        next_values.append(corpus[i + memory].flatten())
    print('nb sequences:', len(sentences))

    print('Creating input...')

    X = np.array(sentences, dtype=np.float32)
    y = np.array(next_values, dtype=np.float32)

    # build the model

    print ("X.shape", X.shape, "y.shape", y.shape)

    print('Build model...')
    model = Sequential()
    # fairly arbitrary layers
    model.add(Dense(X.shape[1]/2, input_dim = X.shape[1], init = 'uniform', activation = 'sigmoid'))
    model.add(Dense(X.shape[1]/4, init='uniform', activation='sigmoid'))
    model.add(Dense(y.shape[1], init='uniform', activation='sigmoid'))

    optimizer = RMSprop(lr=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    # train the model, output generated text after each iteration
    for iteration in range(1, epochs):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=batch_size, nb_epoch=1)

        if (callback_step and iteration%callback_step == 0):

            callback(iteration, model, last = False)
    callback(epochs, model, last = True)
    
    return model