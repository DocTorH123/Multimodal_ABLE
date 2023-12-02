import torch
from PIL import Image
import numpy as np

from os import listdir
import os.path as osp

import torchvision.transforms as transforms

ARTEMIS_EMOTIONS = ['amusement(유희 - 긍정)', 'awe(경외심 - 긍정)', 'contentment(안정감 - 긍정)', 'excitement(흥분감 - 긍정)',
                    'anger(분노 - 부정)', 'disgust(역겨움 - 부정)',  'fear(공포 - 부정)', 'sadness(슬픔 - 부정)', 'something else(이외)']

EMOTION_TO_IDX = {e: i for i, e in enumerate(ARTEMIS_EMOTIONS)}
IDX_TO_EMOTION = {EMOTION_TO_IDX[e]: e for e in EMOTION_TO_IDX}

image_net_mean = [0.485, 0.456, 0.406]
image_net_std = [0.229, 0.224, 0.225]

def image_transformation(img_dim, lanczos=True):
    if lanczos:
        resample_method = Image.LANCZOS
    else:
        resample_method = Image.BILINEAR

    normalize = transforms.Normalize(mean=image_net_mean, std=image_net_std)
    return transforms.Compose([transforms.Resize((img_dim, img_dim), resample_method), transforms.ToTensor(), normalize])

def print_out_emotions(emotion_histogram) :
    max_emotion_idx = 0
    for i in range(len(emotion_histogram)) :
        print(ARTEMIS_EMOTIONS[i], ": ", emotion_histogram[i])
        if emotion_histogram[max_emotion_idx] <= emotion_histogram[i] : max_emotion_idx = i
    print("Max emotion :", ARTEMIS_EMOTIONS[max_emotion_idx], ", Value :", emotion_histogram[max_emotion_idx])

img_dim = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_transformer = image_transformation(img_dim)
weight_path = "Artemis_weights/emotion_classification_best_model.pt"
model = torch.load(weight_path, map_location=None).to(device)

test_img_dir = "../test_images"
test_img_names = [f for f in listdir(test_img_dir)]

for test_img_name in test_img_names :
    test_img = Image.open(osp.join(test_img_dir, test_img_name)).convert('RGB')
    test_img_tensor = image_transformer(test_img).to(device)
    test_img_tensor = test_img_tensor.unsqueeze(0)

    result = model(test_img_tensor)
    result = torch.exp(result).cpu()
    predicted_emotion_histogram = result.detach().numpy()

    for img_idx in range(len(predicted_emotion_histogram)) :
        print_out_emotions(predicted_emotion_histogram[img_idx])


