from PIL import Image

# Label tokenization function
def tokenization_fn(captions, max_target_length, tokenizer) :
    labels = tokenizer(captions, padding="max_length", max_length=max_target_length).input_ids
    return labels

# Image feature extraction function
def feature_extraction_fn(image_paths, feature_extractor) :
    images = [Image.open(image_file) for image_file in image_paths]

    encoder_inputs = feature_extractor(images=images, return_tensors="np")
    return encoder_inputs.pixel_values

# Preprocess function
def preprocess_fn(dataset, max_target_length, label_subject, tokenizer, feature_extractor) :
    image_paths = dataset['image_path']
    captions = dataset[label_subject]

    model_inputs = {'pixel_values': feature_extraction_fn(image_paths, feature_extractor),
                    'labels': tokenization_fn(captions, max_target_length, tokenizer)}

    return model_inputs