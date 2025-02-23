# predict.py
import argparse
from pipeline import Pipeline
from PIL import Image
import numpy as np

def load_image(image_path):
    with Image.open(image_path) as img:
        # Преобразование изображения в формат, подходящий для модели
        img = img.resize((224, 224))  # Предполагаем, что модель ожидает изображения 224x224
        img_array = np.array(img) / 255.0  # Нормализация
        return np.expand_dims(img_array, axis=0)  # Добавление размерности батча

def main(text, image_path, class_to_animal_map, ner_model_path, image_model_path):
    # Инициализация пайплайна
    pipeline = Pipeline(len(class_to_animal_map), class_to_animal_map, ner_model_path, image_model_path)
    
    # Загрузка изображения
    image = load_image(image_path)
    
    # Проверка соответствия текста изображению
    result = pipeline.validate_input(text, image)
    print(f"The text matches the image: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if text description matches the image")
    parser.add_argument("--text", type=str, required=True, help="Text description of the image")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--ner_model", type=str, default="models/ner_model", help="Path to the trained NER model")
    parser.add_argument("--image_model", type=str, default="models/image_model", help="Path to the trained image classification model")
    args = parser.parse_args()
    
    # Пример маппинга классов на животных
    class_to_animal_map = {0: 'cat', 1: 'dog', 2: 'bird', 3: 'horse', 4: 'sheep',
                           5: 'cow', 6: 'elephant', 7: 'bear', 8: 'zebra', 9: 'giraffe'}
    
    main(args.text, args.image, class_to_animal_map, args.ner_model, args.image_model)
