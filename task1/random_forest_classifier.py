# random_forest_classifier.py
from sklearn.ensemble import RandomForestClassifier
from mnist_classifier_interface import MnistClassifierInterface

class RandomForestMnistClassifier(MnistClassifierInterface):
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators)

    def train(self, X_train, y_train):
        self.model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    def predict(self, X_test):
        return self.model.predict(X_test.reshape(X_test.shape[0], -1))


