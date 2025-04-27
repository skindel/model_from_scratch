import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import pyplot

from model import Model
from layers import Layer_Dense
from activation import ReLU, Sigmoid, Softmax

from keras.datasets import mnist
from PIL import Image

import sys


(train_X, train_y), (test_X, test_y) = mnist.load_data()

plt.plot()
plt.imshow(train_X[0], cmap=pyplot.get_cmap('gray'))
plt.show()

# ml = Model(layers=[Layer_Dense(2,4),
#                    Layer_Dense(4,6),
#                    Layer_Dense(6,1)], 
#            activation=Softmax())

# ml.addTrainSet(X, y)

# ml.predict(X)
# ml.calculateLoss()
# print(ml.output)
# print(ml.loss)