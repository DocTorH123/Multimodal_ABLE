from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)

import datasets

# Function for loading model, feature extractor and tokenizer
def load_model(model_path, **kwargs):
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    model.config.decoder.max_length = kwargs["max_length"]
    model.config.decoder.min_length = kwargs["min_length"]
    model.config.decoder.num_beams = kwargs["num_beams"] if "num_beams" in kwargs else 1
    return model

def load_feature_extractor(model_path):
    feature_extractor = ViTImageProcessor.from_pretrained(model_path)
    return feature_extractor

def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer

# Function for loading dataset and adding image_path column to it
def load_dataset(dataset_path, split, cache_dir):
    dataset = datasets.load_dataset(dataset_path, split=split, cache_dir=cache_dir)
    image_path_column = ["./Images/" + str(dataset['ID'][i]) + ".jpg" for i in range(len(dataset))]
    dataset = dataset.add_column("image_path", image_path_column)
    return dataset