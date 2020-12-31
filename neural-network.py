
import numpy as np
from keras.datasets import mnist

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

# learning rate / step size
alpha = 0.01

# train_X = np.array([[0, 1],
#                     [0, 1]])

# train_Y = np.array([[0, 1]])

m = len(train_y)

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
    a[0] = 1 / (1 + np.exp(-z[0]))

    # hidden layer to hidden layer
    for layer in range(hidden_layers-1):
        z[layer+1] = w_hidden_layers[layer].T.dot(a[layer]) + b[layer+1]
        a[layer+1] = 1 / (1 + np.exp(-z[layer+1]))

    # final hidden layer to output layers
    z[-1] = w_output_layer.T.dot(a[-2]) + b[-1]
    a[-1] = 1 / (1 + np.exp(-z[-1]))

    # backward pass
    # output layer to last hidden layer

    da[-1] = -train_y/a[-1] + (1-train_y)/(1-a[-1])
    dz[-1] = da[-1]*(a[-1]*(1-a[-1]))
    dw[-1] = 1/m*dz[-1].dot(a[-2].T)
    db[-1] = np.sum(1/m*np.sum(dz[-1], axis=1, keepdims=True), axis =0, keepdims=True)

    da[-2] = w_output_layer.dot(dz[-1])
    dz[-2] = da[-2]*(a[-2]*(1-a[-2]))
    dw[-2] = 1/m*dz[-2].dot(a[-3].T)
    db[-2] = np.sum(1/m*np.sum(dz[-2], axis=1, keepdims=True), axis =0, keepdims=True)

    for j in range(3, hidden_layers+1):
        da[-j] = w_hidden_layers[-j+2].dot(dz[-j+1])
        dz[-j] = da[-j]*(a[-j]*(1-a[-j]))
        dw[-j] = 1/m*dz[-j].dot(a[-j-1].T)
        db[-j] = np.sum(1/m*np.sum(dz[-j], axis=1, keepdims=True), axis =0, keepdims=True)

    da[0] = w_hidden_layers[0].dot(dz[1])
    dz[0] = da[0]*(a[0]*(1-a[0]))
    dw[0] = 1/m*dz[0].dot(train_X.T)
    db[0] = np.sum(1/m*np.sum(dz[0], axis=1, keepdims=True), axis =0, keepdims=True)

w_input_layer -= alpha * dw[0].T
for i in range(len(w_hidden_layers)):
    w_hidden_layers[i] -= alpha * dw[i+1]
w_output_layer -= alpha * dw[-1].T

b[0] -= alpha * db[0][0]
for i in range(len(w_hidden_layers)):
    b[i+1] -= alpha * db[i+1][0]
b[-1] -= alpha * db[-1][0]

# forward pass
# input layer to first hidden layer
z[0] = w_input_layer.T.dot(test_X) + b[0]
a[0] = 1 / (1 + np.exp(-z[0]))

# hidden layer to hidden layer
for layer in range(hidden_layers-1):
    z[layer+1] = w_hidden_layers[layer].T.dot(a[layer]) + b[layer+1]
    a[layer+1] = 1 / (1 + np.exp(-z[layer+1]))

# final hidden layer to output layers
z[-1] = w_output_layer.T.dot(a[-2]) + b[-1]
a[-1] = 1 / (1 + np.exp(-z[-1]))

y_hat = a[-1]

y_hat[0][np.where(y_hat<0.5)[1]]=0
y_hat[0][np.where(y_hat>0.5)[1]]=1

outcomes = y_hat == test_y

print([ele for ele in y_hat[0]])
print(test_y)


print(f'''Correct: {np.sum(outcomes)}''')
print(f'''Incorrect: {len(outcomes[0]) - np.sum(outcomes)}''')
print(f'''Total: {len(outcomes[0])}''')


# pretty print element to see number
# print('\n'.join(['   '.join([str(cell) for cell in row]) for row in train_X[1]]))
