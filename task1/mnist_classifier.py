# mnist_classifier.py
from random_forest_classifier import RandomForestMnistClassifier
from nn_classifier import FeedForwardNeuralNetwork
from cnn_classifier import ConvolutionalNeuralNetwork

class MnistClassifier:
    def __init__(self, algorithm):
        if algorithm == 'rf':
            self.classifier = RandomForestMnistClassifier()
        elif algorithm == 'nn':
            self.classifier = FeedForwardNeuralNetwork()
        elif algorithm == 'cnn':
            self.classifier = ConvolutionalNeuralNetwork()
        else:
            raise ValueError("Unsupported algorithm. Choose 'rf', 'nn', or 'cnn'.")

    def train(self, X_train, y_train):
        self.classifier.train(X_train, y_train)

    def predict(self, X_test):
        return self.classifier.predict(X_test)
