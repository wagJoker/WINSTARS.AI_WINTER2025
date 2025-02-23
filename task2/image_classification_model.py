
# image_classification_model.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class ImageClassificationModel:
    def __init__(self, num_classes):
        base_model = MobileNetV2(weights='imagenet', include_top=False)
        x = GlobalAveragePooling2D()(base_model.output)
        output = Dense(num_classes, activation='softmax')(x)
        self.model = Model(inputs=base_model.input, outputs=output)

    def train(self, train_data, train_labels, epochs=10):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(train_data, train_labels, epochs=epochs)

    def predict(self, image):
        return self.model.predict(image)
