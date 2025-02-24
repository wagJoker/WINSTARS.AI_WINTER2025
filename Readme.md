Winstars AI DS Internship Test
=====================================

Overview
This project evaluates skills in Data Science, Machine Learning, Computer Vision, and Natural Language Processing (NLP). It consists of two main tasks:

Image Classification + OOP

Utilize the MNIST dataset to build three classification models:

Random Forest

Feed-Forward Neural Network

Convolutional Neural Network

Implement each model as a separate class adhering to the MnistClassifierInterface

Provide a wrapper class MnistClassifier for a unified interface

Named Entity Recognition + Image Classification

Create a pipeline that:

Extracts animal names from text using NER

Classifies animal images

Verifies if the text description matches the image content

Project Structure
text
project/
├── task1/
│   ├── mnist_classifier_interface.py
│   ├── random_forest_classifier.py
│   ├── nn_classifier.py
│   ├── cnn_classifier.py
│   └── mnist_classifier.py
├── task2/
│   ├── train_ner.py
│   ├── image_classifier.py
│   ├── pipeline.py
│   └── requirements.txt
├── images/  # Folder for test images
└── README.md
Setup and Running
Task 1: MNIST Classification
Install dependencies:

bash
pip install -r task1/requirements.txt
Use the models:

python
from mnist_classifier import MnistClassifier

classifier = MnistClassifier(algorithm='cnn')
classifier.train(X_train, y_train)
predictions = classifier.predict(X_test)
Task 2: Animal Verification Pipeline
Install dependencies:

bash
pip install -r task2/requirements.txt
Train models:

bash
python task2/train_ner.py --dataset <ner_dataset_path>
python task2/train_image_classifier.py --dataset <image_dataset_path>
Test the pipeline:

python
from pipeline import AnimalVerificationPipeline

pipeline = AnimalVerificationPipeline()
result = pipeline.verify("This is a cat.", "images/cat.jpg")
print(f"Verification result: {result}")
Testing
Use images in the images/ folder for testing.

Verify text-image pairs using the pipeline.

Requirements
Python 3.8+

See requirements.txt in each task folder for specific libraries.

Notes
Ensure MNIST dataset is available for Task 1.

Prepare a dataset with at least 10 animal classes for Task 2's image classifier.  
