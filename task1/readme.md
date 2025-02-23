This task implements three different classifiers for the MNIST dataset:

Random Forest

Feed-Forward Neural Network

Convolutional Neural Network

Each model is implemented as a separate class that adheres to the MnistClassifierInterface. The MnistClassifier class serves as a wrapper, allowing the user to choose the algorithm at runtime.

How to use:
Train a model:

text
python train.py --algorithm [rf|nn|cnn] --model_path [path_to_save_model]
Make predictions:

text
python predict.py --algorithm [rf|nn|cnn] --model_path [path_to_saved_model]