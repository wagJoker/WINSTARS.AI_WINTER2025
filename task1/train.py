# train.py
import argparse
import os
import pickle
from mnist_classifier import MnistClassifier
from tensorflow.keras.datasets import mnist

def main(algorithm, model_path):
    # Загрузка и нормализация данных
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Создание и обучение классификатора
    classifier = MnistClassifier(algorithm)
    classifier.train(X_train, y_train)

    # Оценка точности на тестовом наборе
    predictions = classifier.predict(X_test)
    accuracy = (predictions == y_test).mean()
    print(f"Test accuracy: {accuracy:.4f}")

    # Сохранение модели
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    
    if algorithm == 'rf':
        with open(model_path, 'wb') as f:
            pickle.dump(classifier.classifier, f)
    else:
        classifier.classifier.model.save(model_path)
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MNIST classifier")
    parser.add_argument("--algorithm", type=str, choices=['rf', 'nn', 'cnn'], required=True,
                        help="Choose the algorithm: Random Forest (rf), Neural Network (nn), or CNN (cnn)")
    parser.add_argument("--model_path", type=str, default="models/mnist_model",
                        help="Path to save the trained model")
    args = parser.parse_args()
    main(args.algorithm, args.model_path)
