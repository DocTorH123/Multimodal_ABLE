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
# torch.nn.Module.dump_patches = True

root_path = "./"
test_img_dir = root_path + "/test images"
test_img_names = [f for f in os.listdir(test_img_dir) if os.path.isfile(os.path.join(test_img_dir, f))]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

temperature = 0.2
title_max_length = 150
overview_max_length = 1024

### Load emotion classification model
print("Loading emotion classification model... ", end="")
image_dim = 256
classifier_image_transformer = classifier_util.classif_image_transformation(image_dim)
classifier_weight_path = root_path + "/Emotion_classifier/Artemis_weights/emotion_classification_best_model.pt"
classifier_model = torch.load(classifier_weight_path, map_location=None).to(device)
print("Success!")

### Load image description model (Title model, Overview description model)
print("Loading image description model... ", end="")
description_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

title_model_path = root_path + "/Descriptor/Title model/best_model"
title_model = VisionEncoderDecoderModel.from_pretrained(title_model_path, local_files_only=True).to(device)
title_feature_extractor = ViTImageProcessor.from_pretrained(title_model_path, local_files_only=True)

overview_model_path = root_path + "/Descriptor/Overview Description model/best_model"
overview_model = VisionEncoderDecoderModel.from_pretrained(overview_model_path, local_files_only=True).to(device)
overview_feature_extractor = ViTImageProcessor.from_pretrained(overview_model_path, local_files_only=True)
print("Success!")

## Make test image batch
print("Preprocessing test images... ", end="")
emotion_image_batch = []
title_image_batch = []
overview_image_batch = []
for test_img_name in test_img_names:
    # Load test image using PIL
    test_img = Image.open(os.path.join(test_img_dir, test_img_name)).convert('RGB')

    # Preprocess test image for each batch
    emotion_image_batch.append(classifier_image_transformer(test_img))
    title_image_batch.append(title_feature_extractor(images=test_img, return_tensors="pt").pixel_values.squeeze(0))
    overview_image_batch.append(overview_feature_extractor(images=test_img, return_tensors="pt").pixel_values.squeeze(0))

emotion_image_batch = torch.stack(emotion_image_batch, dim=0).to(device)
title_image_batch = torch.stack(title_image_batch, dim=0).to(device)
overview_image_batch = torch.stack(overview_image_batch, dim=0).to(device)
print("Success!")

### Inference emotion, title and overview description
print("Inference emotion, title and overview description... ")
print(" * Inference emotion... ", end="")
emotion_results = torch.exp(classifier_model(emotion_image_batch)).cpu()
predicted_emotion_histograms = emotion_results.detach().numpy()
print("Success!")

print(" * Inference title... ", end="")
title_results = title_model.generate(title_image_batch, temperature=temperature, max_length=title_max_length)
predicted_titles = description_tokenizer.batch_decode(title_results, skip_special_tokens=True)
print("Success!")

print(" * Inference overview... ", end="")
overview_results = overview_model.generate(overview_image_batch, temperature=temperature, max_length=overview_max_length)
predicted_overviews = description_tokenizer.batch_decode(overview_results, skip_special_tokens=True)
print("Success!")

print("Fully Success!")

### Print and save results
test_result_path = root_path + "/test images/inferred_results"
if not os.path.exists(test_result_path): os.makedirs(test_result_path)
for i in range(len(test_img_names)):
    # Print results
    print("===============================================================================================")
    print("Image name :", test_img_names[i])
    print("Emotion histogram :")
    max_emotion = classifier_util.print_out_emotions(predicted_emotion_histograms[i])
    print("Title :", predicted_titles[i])
    print("Overview :", predicted_overviews[i])
    print("===============================================================================================")

    # Save results
    print("Saving results " + test_img_names[i] + " ... ", end="")
    with open(test_result_path + "/" + test_img_names[i] + ".txt", "w", encoding="UTF-8") as f:
        # f.write("Image path : " + test_img_dir + "/" + test_img_names[i] + "\n\n")
        # f.write("Image name : " + test_img_names[i] + "\n\n")
        f.write("Title : " + predicted_titles[i] + "\n")
        f.write("Overview Description : " + predicted_overviews[i] + "\n")
        f.write("Emotion histogram : " + str(list(predicted_emotion_histograms[i])) + "\n")
        # f.write("Emotion keyword : " + max_emotion + "\n")
    print("Success!")
