
# ner_model.py
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class NERModel:
    def __init__(self, model_name="dslim/bert-base-NER"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.id2label = self.model.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}

    def train(self, train_data, val_data=None, epochs=3, batch_size=16):
        # Подготовка данных
        train_encodings = self.tokenizer(train_data['texts'], truncation=True, padding=True)
        train_labels = self._encode_labels(train_data['labels'])
        train_dataset = CustomDataset(train_encodings, train_labels)

        if val_data:
            val_encodings = self.tokenizer(val_data['texts'], truncation=True, padding=True)
            val_labels = self._encode_labels(val_data['labels'])
            val_dataset = CustomDataset(val_encodings, val_labels)
        else:
            val_dataset = None

        # Настройка обучения
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # Обучение модели
        trainer.train()

    def _encode_labels(self, labels):
        encoded_labels = []
        for doc_labels in labels:
            doc_enc = []
            for label in doc_labels:
                if label == 'O':
                    doc_enc.append(self.label2id['O'])
                elif label.startswith('B-'):
                    doc_enc.append(self.label2id['B-MISC'])
                elif label.startswith('I-'):
                    doc_enc.append(self.label2id['I-MISC'])
                else:
                    doc_enc.append(self.label2id['O'])
            encoded_labels.append(doc_enc)
        return encoded_labels

    def extract_animal(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        animals = []
        for token, prediction in zip(tokens, predictions[0]):
            if self.id2label[prediction.item()].endswith('-MISC'):  # Предполагаем, что MISC - это метка для животных
                animals.append(token)
        
        return " ".join(animals).lower() if animals else None

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load_model(cls, path):
        model = cls(path)
        return model
