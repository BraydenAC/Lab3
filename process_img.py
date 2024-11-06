from safetensors import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from torch import nn
import pandas as pd
import torch
import os


#Normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Load pretrained ResNet
model = models.resnet50(pretrained=True)

#Load Images
def get_image_rep(image):
    image = transform(image)

    #Add a batch dimension
    image = image.unsqueeze(0)

    #Remove final classification layer to get raw feature vector (embedding)
    resnet_feature_extractor = nn.Sequential(*list(model.children())[:-1]) #Last bit removes final layer

    #get image representation we could use as features for our classifier later
    with torch.no_grad():
        image_representation = resnet_feature_extractor(image)

    #Reshape output from [1, 512, 1, 1] to [1, 512]
    image_representation = image_representation.view(image_representation.size(0), -1)

    return image_representation

def process_set(pic_list):
    #Initialize storage dataframe
    processed = pd.DataFrame()

    for i in range(len(pic_list)):
        try:
            print(f"processing image {pic_list[i]}")
            pic = Image.open("Datasets/hateful_memes/" + pic_list[i])

            #Ensuring picture is in right format
            pic = pic.convert("RGB")

            processing = pd.DataFrame(get_image_rep(pic).numpy())
        except FileNotFoundError:
            processing = pd.DataFrame([[float('nan')] * 2048])
            print(f"Image {pic_list[i]} not found!")

        processed = pd.concat([processed, processing], ignore_index=True)
    return processed

#Retrieve the img name lists from sets
training_set = pd.read_json("Datasets/hateful_memes/train.jsonl", lines=True)
training_img_names = training_set["img"].tolist()
dev_set = pd.read_json("Datasets/hateful_memes/dev_seen.jsonl", lines=True)
dev_img_names = dev_set["img"].tolist()
test_set = pd.read_json("Datasets/hateful_memes/test_seen.jsonl", lines=True)
test_img_names = test_set["img"].tolist()

#Process each list
print("Processing training set...")
processed_train = process_set(training_img_names)
print("Processing dev set...")
processed_dev = process_set(dev_img_names)
print("Processing test set...")
processed_test = process_set(test_img_names)

#Save the processed data
print("Saving data to img_train.csv...")
processed_train.to_csv('Datasets/hateful_memes/img_train.csv', index=False)
print("Saving data to img_dev.csv...")
processed_train.to_csv('Datasets/hateful_memes/img_dev.csv', index=False)
print("Saving data to img_test.csv...")
processed_train.to_csv('Datasets/hateful_memes/img_test.csv', index=False)
print("Done!")