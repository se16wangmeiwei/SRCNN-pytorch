# from __future__ import print_function

import argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from predict import imshow

from demo import SRCNNTrainer
from data import get_training_set, get_test_set, get_predict_img
from data import get_predict_img
from predict import predict

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--model', '-m', type=str, default='srcnn', help='choose which model is going to use')


args = parser.parse_args()
print(args)


def main():
    # ===========================================================
    # Set train dataset & test dataset
    # ===========================================================
    print('===> Loading datasets')
    train_set = get_training_set(args.upscale_factor)
    test_set = get_test_set(args.upscale_factor)

    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=False)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)

    # if args.model == 'sub':
    #     model = SubPixelTrainer(args, training_data_loader, testing_data_loader)
    if args.model == 'srcnn':
        model = SRCNNTrainer(args, training_data_loader, testing_data_loader)
    # elif args.model == 'vdsr':
    #     model = VDSRTrainer(args, training_data_loader, testing_data_loader)
    # elif args.model == 'edsr':
    #     model = EDSRTrainer(args, training_data_loader, testing_data_loader)
    # elif args.model == 'fsrcnn':
    #     model = FSRCNNTrainer(args, training_data_loader, testing_data_loader)
    # elif args.model == 'drcn':
    #     model = DRCNTrainer(args, training_data_loader, testing_data_loader)
    # elif args.model == 'srgan':
    #     model = SRGANTrainer(args, training_data_loader, testing_data_loader)
    # elif args.model == 'dbpn':
    #     model = DBPNTrainer(args, training_data_loader, testing_data_loader)
    # else:
    #     raise Exception("the model does not exist")

    model.run()


if __name__ == '__main__':
    # device = torch.device('cpu')
    # main()
    # loader使用torchvision中自带的transforms函数
    input_img, result_img = predict("../test_data/Set14/baboon.bmp")
    plt.figure()
    plt.subplot(1, 2, 1)
    imshow(input_img, 'input image')
    plt.subplot(1, 2, 2)
    imshow(result_img, 'result image')
    plt.show()
