import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image
from DAMSLNet import DAMSLNet


if __name__ == "__main__":

    device = torch.device("cuda")
    checkpoint = torch.load('./best_model.pth')
    model = DAMSLNet(12).to(device)
    model.load_state_dict(checkpoint)

    data_transform = transforms.Compose([
                                        transforms.ToTensor()
                                        ])

    img = "./image_081.jpg"
    img = Image.open(img)
    img = transforms.Resize((256,256))(img)
    img_np = np.array(img, dtype=np.uint8)
    img_tensor = data_transform(img).to(device)
    img_tensor_1 = torch.unsqueeze(img_tensor,dim=0)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    # Output of icpA layer
    ######################################################################
    target_layers = [model.icpA]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    target_category = 0

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img_np.astype(dtype=np.float32) / 255.,
                                        grayscale_cam,
                                        use_rgb=True)
    plt.imshow(visualization)
    plt.savefig("icpA.png")


    # Output of redA layer
    ######################################################################
    target_layers = [model.redA]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    target_category = 0

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img_np.astype(dtype=np.float32) / 255.,
                                        grayscale_cam,
                                        use_rgb=True)
    plt.imshow(visualization)
    plt.savefig("redA.png")


    # Output of icpB layer
    ######################################################################
    target_layers = [model.icpB]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    target_category = 0

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img_np.astype(dtype=np.float32) / 255.,
                                        grayscale_cam,
                                        use_rgb=True)
    plt.imshow(visualization)
    plt.savefig("icpB.png")


    # Output of redBlayer
    ######################################################################
    target_layers = [model.redB]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    target_category = 0

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img_np.astype(dtype=np.float32) / 255.,
                                        grayscale_cam,
                                        use_rgb=True)
    plt.imshow(visualization)
    plt.savefig("redB.png")



    # Output of DA layer
    ######################################################################
    target_layers = [model.decoder]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    target_category = 0

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img_np.astype(dtype=np.float32) / 255.,
                                        grayscale_cam,
                                        use_rgb=True)
    plt.imshow(visualization)
    plt.colorbar(plt.cm.ScalarMappable(cmap='jet'),ax=plt.gca(),orientation='vertical')
    plt.savefig("DA Output.png")
