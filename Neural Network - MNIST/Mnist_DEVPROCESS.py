# https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py
# https://nextjournal.com/schmudde/ml4a-mnist
# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py (CONVOLUTIONAL)
# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/01_Simple_Linear_Model.ipynb (TF)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
import random

'''
TRY:
-> Loss functions:
Log loss 
KL Divergence
Mean Squared Error
Categorical_crossentropy (atual)

-> Optimizers:
RMSprop (atual)


'''


batch_size = 128
num_classes = 10 # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} = 10
epochs = 20

(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.figure()
numbers = np.concatenate([np.concatenate([x_train[i] for i in [int(random.random() * len(x_train)) for i in range(9)]], axis=1) for i in range(9)], axis=0)
plt.imshow(numbers, cmap='gist_gray', interpolation='none')
plt.xticks([])
plt.yticks([])
plt.xlabel('Alguns números do MNIST')
plt.show()

'''
r = int(random.random() * len(x_train))
numbers2 = np.array([x_train[x] for x in range(r-5, r)])
plt.xlabel('Alguns números do MNIST, começando de ' + str(r-5))
plt.imshow(numbers2.reshape(int(numbers2.size/28), 28), cmap='gist_gray')
'''

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255 # Cores [0, 255] -> [0, 1]
x_test /= 255 

print(x_train.shape[0], 'números/arquivos de treino')
print(x_test.shape[0], 'números/arquivos de teste')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

classifier = Sequential()
classifier.add(Dense(512, activation='relu', input_shape=(784,)))
classifier.add(Dropout(0.2))
classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(num_classes, activation='softmax'))

#classifier.summary()

classifier.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

classifier.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = classifier.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])