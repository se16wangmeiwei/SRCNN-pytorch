import torch
from data import get_predict_img
# from PIL import Image
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    # plt.pause(10)  # pause a bit so that plots are updated


def predict(img_path):
    net = torch.load('model_path.pth')
    net = net.to(device)
    torch.no_grad()
    input_img, target_img = get_predict_img(4, img_path)
    input_img = input_img.unsqueeze(1)
    outputs = net(input_img)
    return input_img, outputs


device = torch.device('cpu')




