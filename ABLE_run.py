import sys
import os

import numpy
import torch
from PIL import Image

from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)

import Emotion_classifier.util_functions as classifier_util

### Shared variable for ABLE inference
root_path = "./Multimodal_ABLE"
test_img_dir = root_path + "/test_images"
test_img_names = [f for f in os.listdir(test_img_dir)]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Load emotion classification model
image_dim = 256
classifier_image_transformer = classifier_util.classif_image_transformation(image_dim)
classifier_weight_path = root_path + "/Emotion_classifier/Artemis_weights/emotion_classification_best_model.pt"
classifier_model = torch.load(classifier_weight_path, map_location=None).to(device)

### Load image description model (Title model, Overview description model)
description_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

title_model_path = root_path + "/Title model/best_model"
title_model = VisionEncoderDecoderModel.from_pretrained(title_model_path, local_files_only=True).to(device)
title_feature_extractor = ViTImageProcessor.from_pretrained(title_model_path, local_files_only=True)

overview_model_path = root_path + "/Overview model/best_model"
overview_model = VisionEncoderDecoderModel.from_pretrained(overview_model_path, local_files_only=True).to(device)
overview_feature_extractor = ViTImageProcessor.from_pretrained(overview_model_path, local_files_only=True)

## Make test image batch
emotion_image_batch = []
test_images = []
for test_img_name in test_img_names:
    # Load test image using PIL
    test_img = Image.open(os.path.join(test_img_dir, test_img_name)).convert('RGB')
    test_images.append(test_img)

    # Preprocess test image for emotion classification
    emotion_image_batch.append(classifier_image_transformer(test_img))

# Convert each batch to torch tensor and move to device
test_images = numpy.asarray(test_images)
emotion_image_batch = torch.stack(emotion_image_batch).to(device)
title_image_batch = title_feature_extractor(images=test_images, return_tensors="pt").pixel_values.to(device)
overview_image_batch = overview_feature_extractor(images=test_images, return_tensors="pt").pixel_values.to(device)

### Inference emotion, title and overview description
emotion_results = torch.exp(classifier_model(emotion_image_batch)).cpu()
predicted_emotion_histograms = emotion_results.detach().numpy()

title_results = title_model.generate(title_image_batch)
predicted_titles = description_tokenizer.batch_decode(title_results, skip_special_tokens=True)

overview_results = overview_model.generate(overview_image_batch)
predicted_overviews = description_tokenizer.batch_decode(overview_results, skip_special_tokens=True)

### Print and save results
test_result_path = root_path + "/test_images/results"
if not os.path.exists(test_result_path): os.makedirs(test_result_path)
for i in range(len(test_img_names)):
    # Print results
    print("Image name :", test_img_names[i])
    print("Emotion histogram :", predicted_emotion_histograms[i])
    print("Title :", predicted_titles[i])
    print("Overview :", predicted_overviews[i])

    # Save results
    with open(test_result_path + "/" + test_img_names[i] + ".txt", "w") as f:
        f.write("Image path : " + test_img_dir + "/" + test_img_names[i] + "\n\n")
        f.write("Image name : " + test_img_names[i] + "\n\n")
        f.write("Emotion histogram : " + str(predicted_emotion_histograms[i]) + "\n\n")
        f.write("Title : " + predicted_titles[i] + "\n\n")
        f.write("Overview : " + predicted_overviews[i] + "\n\n")