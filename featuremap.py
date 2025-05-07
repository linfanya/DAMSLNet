import torch
import torch.nn as nn
import numpy as np
from DAMSLNet import DAMSLNet
import torchvision.transforms as transforms
# from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt


# Loading the network model
checkpoint = torch.load('./best_model.pth')
model = DAMSLNet(12)
model.load_state_dict(checkpoint)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_path = './image (3).JPG'  # Replace with your image path
img = Image.open(img_path)
img_tensor = transform(img).unsqueeze(0)


# Extracting feature maps
def get_features(module, input, output):
    features.append(output.detach())  # save the output feature maps

features = []
layer = model.icpA.tb.conv # Replace 'layer' with the specific layer you want to extract feature maps from
layer.register_forward_hook(get_features)

# Forward pass
with torch.no_grad():
    model(img_tensor)  # pass the image through the model

# Visualizing the feature maps
def plot_feature_maps(feature_maps):
    num_features = feature_maps.shape[1]
    plt.figure(figsize=(15, 15))
     # Adjust the layout according to the number of extracted feature maps, here we take 8x8 as an example
    for i in range(min(num_features, 64)):
        ax = plt.subplot(8, 8, i + 1) 
        plt.imshow(feature_maps[0, i], cmap='viridis')
        plt.axis('off')
    plt.show()
    plt.savefig("featuremap.jpg")
plot_feature_maps(features[0])  # Display the first layer feature map

