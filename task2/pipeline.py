# pipeline.py
from ner_model import NERModel
from image_classification_model import ImageClassificationModel

class Pipeline:
    def __init__(self, num_classes, class_to_animal_map):
        self.ner_model = NERModel()
        self.image_model = ImageClassificationModel(num_classes)
        self.class_to_animal_map = class_to_animal_map

    def validate_input(self, text_input, image_input):
        animal_name = self.ner_model.extract_animal(text_input)
        
        if animal_name:
            predicted_class = self.image_model.predict(image_input).argmax()
            predicted_animal = self.class_to_animal_map.get(predicted_class)
            
            return predicted_animal == animal_name
        
        return False

