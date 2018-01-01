import numpy as np
import matplotlib.pyplot as plt
from util import get_data, softmax, cost2, y2indicator, error_rate, relu
from sklearn.utils import shuffle

class Ann:


	def __init__(self, M):
		self.M = M

	def fit (self, X, Y, learning_rate=1e-6, reg=1e-6, epochs=10000, show_fig=False):
		X, Y = shuffle(X, Y)
		Xvalid, Yvalid = X[-1000:], Y[-1000:]
		X, Y = X[:1000], Y[:1000]

		N, D = X.shape
		K = max(Y) + 1 
		T = y2indicator(Y)
		self.W1 = np.random.randn(D, self.M) / np.sqrt(D)
		self.b1 = np.zeros(self.M)
		self.W2 = np.random.randn(self.M, k) / np.sqrt(self.M)
		self.b2 = np.zeros(K)

		cost = []
		best_validation_error = 1

		# forward propagation and cost calculation
		for i in range(epochs):
			pY, Z = self.forward(X)

			# gradient descent step
			pY_T = pY - T
			self.W2 -= learning_rate * (Z.T.dot(pY_T) + reg * self.W2)
			self.b2 -= learning_rate * (pY_T.sum(axis=0) + reg * self.b2)
			# dZ = pY_T.dot(self.W2.T) * (Z > 0) #relu
			dZ = pY_T.dot(self.W2.T) * (1 - Z * Z) # tanh
			self.W1 -= learning_rate * (X.T.dot(Z) + reg * self.W1)
			self.b1 -= learning_rate * (dZ.sum(axis=0) + reg * self.b1)

			if i % 10 == 0:
				pYvalid, _ = self.forward(Xvalid)
				c = cost2(Yvalid, pYvalid)
				costs.append(c)
				e = error_rate(Yvalid, np.argmax(pYvalid, axis=1))
				print('i: ', i, 'cost:', c, 'error: ', e)
				if e < best_validation_error:
					best_validation_error = e
		print('best_validation_error: ', best_validation_error)

		if show_fig:
			plt.plot(costs)
			plt.show()


	def forward(self, X):
		# Z is N by M matrix, return a N by K matrix
		# Z = relu(X.dot(self.W1) + self.b1)
		Z = np.tanh(X.dot(self.W1) + self.b1)
		return softmax(Z.dot(self.W2) + self.b2), Z

	def predict(self, X):
		pY, _ = self.forward(X)
		return np.argmax(pY, axis=1)

	def score(self, X, Y):
		prediction = self.predict(X)
		return 1 - error_rate(Y, prediction)

def main():
	X, Y = get_data()
	model = Ann(200)
	model.fit(X, Y, reg=0, show_fig=True)
	print(model.score(X, Y))
	scores = cross_val_score(model, X, Y, cv=5)
	print('score mean: ', np.mean(scores), 'stdev: ', np.std(scores))


if __name__ == '__main__':
	main()





