from PIL import Image

import torchvision.transforms as transforms

artemis_emotions = ['amusement(유희 - 긍정)', 'awe(경외심 - 긍정)', 'contentment(안정감 - 긍정)', 'excitement(흥분감 - 긍정)',
                    'anger(분노 - 부정)', 'disgust(역겨움 - 부정)', 'fear(공포 - 부정)', 'sadness(슬픔 - 부정)', 'something else(이외)']

image_net_mean = [0.485, 0.456, 0.406]
image_net_std = [0.229, 0.224, 0.225]

def classif_image_transformation(img_dim, lanczos=True):
    if lanczos:
        resample_method = Image.LANCZOS
    else:
        resample_method = Image.BILINEAR

    normalize = transforms.Normalize(mean=image_net_mean, std=image_net_std)
    return transforms.Compose([transforms.Resize((img_dim, img_dim), resample_method), transforms.ToTensor(), normalize])

def print_out_emotions(emotion_histogram) :
    max_emotion_idx = 0
    for i in range(len(emotion_histogram)) :
        print(artemis_emotions[i], ": ", emotion_histogram[i])
        if emotion_histogram[max_emotion_idx] <= emotion_histogram[i] : max_emotion_idx = i
    print("Max emotion :", artemis_emotions[max_emotion_idx], ", Value :", emotion_histogram[max_emotion_idx])

    return artemis_emotions[max_emotion_idx]
