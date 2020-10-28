from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from PIL import Image

from dataset import DatasetFromFolder


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    #
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor):
    # root_dir = download_bsd300()
    print("开始加载训练数据")
    train_dir = "../test_data/B101"
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
    # root_dir = download_bsd300()
    print("开始加载测试数据")
    print(upscale_factor)
    test_dir = "../test_data/B100"
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_predict_img(upscale_factor, pre_dir):
    print("add predict image")
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    input_image = load_img(pre_dir)
    target_image = input_image.copy()
    input_tran = input_transform(crop_size, upscale_factor)
    input_image = input_tran(input_image)
    target_tran = target_transform(crop_size)
    target_image = target_tran(target_image)
    return input_image, target_image
