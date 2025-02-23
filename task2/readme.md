English Description
This task creates a pipeline that combines Named Entity Recognition (NER) for extracting animal names from text and Image Classification for identifying animals in images. The pipeline verifies if the animal mentioned in the text matches the animal in the image.

How to use:
Train NER model:

text
python train_ner.py --dataset [path_to_ner_dataset] --model_path [path_to_save_ner_model]
Train Image Classifier:

text
python train_image_classifier.py --dataset [path_to_image_dataset] --model_path [path_to_save_image_model]
Run the pipeline:

text
python predict.py --text "There is a cat in the image" --image [path_to_image] --ner_model [path_to_ner_model] --image_model [path_to_image_model]