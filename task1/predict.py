# predict.py
import argparse
import numpy as np
import pickle
from mnist_classifier import MnistClassifier
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

def main(algorithm, model_path):
    # Загрузка тестовых данных
    (_, _), (X_test, y_test) = mnist.load_data()
    X_test = X_test / 255.0

    # Создание классификатора и загрузка обученной модели
    classifier = MnistClassifier(algorithm)
    if algorithm == 'rf':
        with open(model_path, 'rb') as f:
            classifier.classifier = pickle.load(f)
    else:
        classifier.classifier.model = load_model(model_path)

    # Выполнение предсказаний
    predictions = classifier.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Вывод нескольких примеров предсказаний
    print("\nSample predictions:")
    for i in range(10):
        print(f"True: {y_test[i]}, Predicted: {predictions[i]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using trained MNIST classifier")
    parser.add_argument("--algorithm", type=str, choices=['rf', 'nn', 'cnn'], required=True,
                        help="Choose the algorithm: Random Forest (rf), Neural Network (nn), or CNN (cnn)")
    parser.add_argument("--model_path", type=str, default="models/mnist_model",
                        help="Path to the trained model")
    args = parser.parse_args()
    main(args.algorithm, args.model_path)
