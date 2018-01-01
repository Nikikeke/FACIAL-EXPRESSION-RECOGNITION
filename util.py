import numpy as np
import pandas as pd

#M1, input size. M2, output size.
def init_weight_bias(M1, M2):
	W = np.random.randn(M1, M2) / np.sqrt(M1)
	b = np.zeros(M2)
	return W.astype(np.float32), b.astype(np.float32)

def init_filter(shape, poolsz):
	kernal = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(np.prod(shape[1:]) + shape[0] * np.prod(shape[2:] / np.prod(poolsz)))
	return kernal.astype(np.float32)

def relu(x):
	return max(0, x)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def softmax(A):
	expA = np.exp(A)
	return expA / expA.sum(axis=1, keepdims=True)

# corss entropy for sigmoid cost, binary classification
def sigmoid_cost(T, Y):
	return -(T * np.log(Y) + (1 - T) * np.log(1 - Y)).sum()

# general cross entropy function, work for softmax
def cost(T, Y):
	return -(T * np.log(Y)).sum()

#same result as cost, just uses the targets to index Y
#instead of multiplying by a large indicator matrix with mostly 0s
def cost2(T, Y):
	N = len(T)
	return -np.log(Y[np.arange(N), T]).mean()

def error_rate(targets, predictions):
	return np.mean(targets != predictions)

# one hot encoding
def y2indicator(y):
	N = len(y)
	K = max(y) + 1  # y ranges from 0 to 6
	ind = np.zeros((N, K))
	for i in range(N):
		ind[i, y[i]] = 1
	return ind

def get_data(balance_ones=True):
    X = []
    Y = []
    with open('/Users/Xueyao/Documents/GitHub/fer2013.csv', 'r') as f:
        next(f, None)
        for line in f:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])
    X, Y = np.array(X) / 255.0, np.array(Y)
    
    # handle imbalanced class, add dummy samples to underrepresented class
    if balance_ones:
    	X0, Y0 = X[Y!=1, :], Y[Y!=1]
    	X1 = X[Y==1, :]
    	X1 = np.repeat(X1, 9, axis=0)
    	X = np.vstack([X0, X1])
    	Y = np.concatenate((Y0, [1]*len(X1)))

    return X, Y

def get_image_data():
	X, Y = get_data()
	N, D = X.shape
	d = int(np.sqrt(D))
	X = X.reshape(N, 1, d, d)
	return X, Y

# split data into k parts
def cross_validation(model, X, Y, K=5):
	X, Y = shuffle(X, Y)
	sz = len(Y) // K
	errors = []
	for k in range(K):
		xtrain = np.concatenate([X[:k*sz, :], X[(k*sz + sz):, :]])
		ytrain = np.concatenate([Y[:k*sz], Y[(k*sz + sz):]])
		xtest = X[k*sz:(k*sz + sz), :]
		ytest = Y[k*sz:(k*sz + sz)]

		model.fit(xtrain, ytrain)
		err = model.score(xtest, ytest)
		errors.append(err)

	print('errors: ', errors)
	return np.mean(errors)
























