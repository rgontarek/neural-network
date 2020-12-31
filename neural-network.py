
import numpy as np
from keras.datasets import mnist

# setup
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_y = train_y.reshape(1, 60000)
test_y = test_y.reshape(1, 10000)
train_X = np.delete(train_X, np.where(train_y > 1), axis=0)
train_X = np.divide(train_X, 255)
train_y = np.delete(train_y, np.where(train_y > 1), axis=1)
test_X = np.delete(test_X, np.where(test_y > 1), axis=0)
test_X = np.divide(test_X, 255)
test_y = np.delete(test_y, np.where(test_y > 1), axis=1)
train_X = train_X.reshape(12665,784).T
test_X = test_X.reshape(2115,784).T

m = len(train_y)

# hyperparameters
alpha = 0.01
hidden_layers = 20 # can't be less than 2 for this implementation
input_nodes = 784
hidden_nodes = 50
output_nodes = 1

w_input_layer = np.random.rand(input_nodes,hidden_nodes)
w_hidden_layers = [np.random.rand(hidden_nodes, hidden_nodes) for i in range(hidden_layers-1)]
w_output_layer = np.random.rand(hidden_nodes,output_nodes)

b = np.random.rand(hidden_layers + 1, 1)

z = [[]]*(hidden_layers+1)
a = [[]]*(hidden_layers+1)

da = [[]]*(hidden_layers+1)
dz = [[]]*(hidden_layers+1)
dw = [[]]*(hidden_layers+1)
db = [[]]*(hidden_layers+1)

for i in range(2):

    # forward pass
    # input layer to first hidden layer
    z[0] = w_input_layer.T.dot(train_X) + b[0]
    
#################################################
# This becomes NaN after second iteration. why? #
#################################################
    a[0] = 1 / (1 + np.exp(-z[0]))
#################################################
# This becomes NaN after second iteration. why? #
#################################################
    
    # hidden layer to hidden layer
    for layer in range(hidden_layers-1):
        z[layer+1] = w_hidden_layers[layer].T.dot(a[layer]) + b[layer+1]
        a[layer+1] = 1 / (1 + np.exp(-z[layer+1]))

    # final hidden layer to output layers
    z[-1] = w_output_layer.T.dot(a[-2]) + b[-1]
    a[-1] = 1 / (1 + np.exp(-z[-1]))
