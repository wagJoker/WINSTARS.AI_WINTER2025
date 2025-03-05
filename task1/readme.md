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


Setting up a Virtual Environment

Create a new virtual environment:

python -m venv venv

Activate the virtual environment and install dependencies:

venv\Scripts\activate   # for Windows
source venv/bin/activate   # for Linux/macOS

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

Task 1: MNIST Classification

Training Models

Run the train.py script for each algorithm:

python train.py --algorithm rf --model_path models/rf_model.pkl
python train.py --algorithm nn --model_path models/nn_model.h5
python train.py --algorithm cnn --model_path models/cnn_model.h5

Prediction and Accuracy Evaluation

Use the predict.py script to evaluate the accuracy of each model on the MNIST test dataset:

python predict.py --algorithm rf --model_path models/rf_model.pkl
python predict.py --algorithm nn --model_path models/nn_model.h5
python predict.py --algorithm cnn --model_path models/cnn_model.h5

Comparing Results

Compare the prediction accuracy of each algorithm and determine which performs best.

