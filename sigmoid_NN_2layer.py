import numpy as np 

def sigmoid (z):
    return 1/(1 + np.exp(-z))

def sigmoid_derivative (a):
    return a(1-a)

def compute_losses (y_true,y_pred):
    m = y_true.shape[0]
    return -np.mean(y_true * np.log(y_pred + 1e - 8) + (1 - y_true)*np.log(1-y_pred + 1e - 8))

#XOR dataset 
X = np.array ([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array ([[0],
              [1],
              [1],
              [0]])

#Network parameters
np.random.seed(42)
n_input = 2 
n_hidden = 2
n_output = 1 
m = X.shape[0]

#Define matrix shape 

