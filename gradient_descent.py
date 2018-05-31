import numpy as np 
import matplotlib.pyplot as plt

class GradientDescent():
	def __init__(self, alpha=0.1, tolerance=0.02, max_iterations=500):
		#alpha is the learning rate or size of step to take in 
		#the gradient decent
		self._alpha = alpha
		self._tolerance = tolerance
		self._max_iterations = max_iterations
		#thetas is the array coeffcients for each term
		#the y-intercept is the last element
		self._thetas = None

	def fit(self, xs, ys):
		num_examples, num_features = np.shape(xs)
		self._thetas = np.ones(num_features)
        
		xs_transposed = xs.transpose()
		for i in range(self._max_iterations):
			#difference between our hypothesis and actual values
			diffs = np.dot(xs,self._thetas) - ys
			#sum of the squares
			cost = np.sum(diffs**2) / (2*num_examples)
			#calculate averge gradient for every example
			gradient = np.dot(xs_transposed, diffs) / num_examples
			#update the coeffcients
			self._thetas = self._thetas-self._alpha*gradient
			
			#check if fit is "good enough"
			if cost < self._tolerance:
				return self._thetas

		return self._thetas

	def predict(self, x):
		return np.dot(x, self._thetas)