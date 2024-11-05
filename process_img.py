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

#Get a list of all image names
print("Getting image name list...")
pic_list = os.listdir("Datasets/hateful_memes/img")

#Initialize storage dataframe
processed = pd.DataFrame()

for i in range(len(pic_list)):
    print(f"processing image {pic_list[i]}")
    pic = Image.open("Datasets/hateful_memes/img/" + pic_list[i])

    #Ensuring picture is in right format
    pic = pic.convert("RGB")

    processing = pd.DataFrame(get_image_rep(pic).numpy())
    processed = pd.concat([processed, processing], ignore_index=True)


#Save the processed data
print("Saving data to img.csv...")
processed.to_csv('Datasets/hateful_memes/img.csv', index=False)
print("Done!")