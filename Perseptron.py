import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate
    
    def predict(self, input_vector):
        linear_output = np.dot(self.weights, input_vector) + self.bias
        prediction = np.where(linear_output > 0, 1, 0)
        return prediction
    
    def train(self, training_inputs, labels):
        for inputs, label in zip(training_inputs, labels):
            prediction = self.predict(inputs)
            error = label - prediction
            self.weights += self.learning_rate * error * inputs
            self.bias += self.learning_rate * error