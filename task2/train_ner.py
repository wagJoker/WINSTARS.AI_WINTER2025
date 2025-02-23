# train_ner.py
import argparse
import json
from ner_model import NERModel

def load_data(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {
        'texts': [item['text'] for item in data],
        'labels': [item['labels'] for item in data]
    }

def main(dataset_path, model_path):
    # Загрузка данных
    train_data = load_data(dataset_path)
    
    # Инициализация и обучение модели
    ner_model = NERModel()
    ner_model.train(train_data)
    
    # Сохранение обученной модели
    ner_model.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NER model for animal detection")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset JSON file")
    parser.add_argument("--model_path", type=str, default="models/ner_model", help="Path to save the trained model")
    args = parser.parse_args()
    main(args.dataset, args.model_path)

