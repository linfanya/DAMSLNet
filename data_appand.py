#######################################################################
# Batch rename image files
#######################################################################
# import os

# # Define folder path
# folder_path = "./scab_frog_eye_leaf_spot"

# # Traverse the image files in the folder
# for i, filename in enumerate(os.listdir(folder_path)):
#     if filename.endswith(".jpg"): #or filename.endswith(".png"):
#         # Construct a new file name, such as image_001.jpg, image_002.jpg, ...
#         new_filename = f"image_{i+1:03d}.jpg"
        
#         # Construct original file path and new file path
#         old_path = os.path.join(folder_path, filename)
#         new_path = os.path.join(folder_path, new_filename)
        
#         # rename the file
#         os.rename(old_path, new_path)
# print("succeffully!")

#######################################################################
# Specify images for data augmentation
#######################################################################
# import os
# from PIL import Image
# from torchvision import transforms

# # Define data augmentation transformation
# data_transform = transforms.Compose([
#     # transforms.RandomHorizontalFlip(p=1.0), # Flip by probability level
#     # transforms.RandomVerticalFlip(p=1.0), # Flip vertically by probability
#     # transforms.RandomRotation(10),#-15° to +15° random rotation
#     # transforms.ColorJitter(brightness=(1.3, 1.301)), # Randomly change the brightness of the image
#     # transforms.ColorJitter(brightness=(0.5, 0.601)), # Randomly change the darkness of the image
#     # transforms.ColorJitter(contrast=(1.3, 1.501)), # Randomly change the contrast of the image
#     # transforms.ColorJitter(saturation=0.2),#Randomly adjust saturation
#     transforms.ColorJitter(hue=0.2),#Randomly adjust hue
# ])

# # Define the path of the original dataset
# data_dir = "./Apple leaf"

# # Traverse the image files in the folder
# for root, dirs, files in os.walk(data_dir):
#     for file in files:
#         # Only process image files
#         if file.endswith(".jpg"):
#             # loading images
#             image_path = os.path.join(root, file)
#             # image = Image.open(image_path)
#             #Convert it to RGB format
#             image = Image.open(image_path).convert('RGB')
            
#             # Application data augmentation
#             augmented_image = data_transform(image)
            
#             # Obtain the path to save the enhanced image
#             save_path = os.path.join(root, file.split(".")[0] + "hue.JPG")
            
#             # save the enhanced image
#             augmented_image.save(save_path)
# print("successfully!")

######################################################################
# data augmentation
# ######################################################################
import albumentations as transforms
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2

# Input output address
# input_folder = 'testdata'
# output_folder = '/kaggle/working/'

# Normalize the pixel values of the image
# normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

train_transform = transforms.Compose([
    # transforms.Resize(256, 256),
    # transforms.RandomFog(fog_coef_lower=0.25, fog_coef_upper=0.6, alpha_coef=0.3, always_apply=True,p=0.6),#Randomly add fog
    transforms.RandomRain(slant_lower=-5, slant_upper=5, 
                          drop_length=5, drop_width=1, drop_color=(200, 200, 200), 
                          blur_value=1, brightness_coefficient=0.6, always_apply=True,p=0.6),#Randomly add rain
    # transforms.RandomShadow(shadow_roi=(0, 0.5, 1, 1), 
    #                         num_shadows_lower=1, num_shadows_upper=5, 
    #                         shadow_dimension=6, always_apply=True,p=0.6),#Randomly add shadow
    # transforms.RandomSunFlare(flare_roi=(0, 0, 1, 0.1), 
    #                           angle_lower=0, angle_upper=1, 
    #                           num_flare_circles_lower=2, num_flare_circles_upper=4,
    #                           src_radius=150, src_color=(255, 255, 255),
    #                           always_apply=True,p=0.6),#Randomly add sunflare
    # transforms.ToTensor(),
    # normalize
    ])

# Define the path of the original dataset
data_dir = "./Tomato Septoria leaf spot"

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".jpg"):
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            augmented = train_transform(image=image)
            augmented_image = augmented['image']
            
            save_path = os.path.join(root, file.split(".")[0] + "Rain.JPG")
            
            cv2.imwrite(save_path, augmented_image)
print("successfully!")

######################################################################
# Delete images with a. jpg suffix from the folder
######################################################################
# import os

# folder_path = "/home/denglinfan/plantclassification/linfan/Apple_FGVC8/train_dataagument_256/scab frog_eye_leaf_spot complex"

# for filename in os.listdir(folder_path):
#     if filename.endswith(".jpg"):
#         file_path = os.path.join(folder_path, filename)
        
#         # delate the file
#         os.remove(file_path)
# print("successfully!")

