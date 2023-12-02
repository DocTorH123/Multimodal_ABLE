import torch
from PIL import Image

from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)

from os import listdir

# Load model, feature extractor and tokenizer from local
title_model_path = "./Title model/best_model"
title_model = VisionEncoderDecoderModel.from_pretrained(title_model_path, local_files_only=True)
title_feature_extractor = ViTImageProcessor.from_pretrained(title_model_path, local_files_only=True)
title_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
title_model.to(device)

# Read test images and inference
test_img_dir = "../test_images"
test_img_names = [f for f in listdir(test_img_dir)]

for test_img_name in test_img_names :
    test_img = Image.open(test_img_dir + "/" + test_img_name).convert('RGB')
    test_img_tensor = title_feature_extractor(images=test_img, return_tensors="pt").pixel_values.to(device)

    # Inference
    result = title_model.generate(test_img_tensor)
    result = title_tokenizer.batch_decode(result, skip_special_tokens=True)

    # Print results
    print("Image name :", test_img_name)
    print("Inference result :", result)
    print()


