import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from argparse import ArgumentParser
from torchvision import datasets, transforms

torch.manual_seed(999)

def gray2bin(image, threshold = -1.0):
    bin_image = (image > threshold).int()
    if threshold >= 0:
        return np.array(bin_image).reshape((28, 28))
    return np.array(image).reshape((28, 28))

def get_mnist():
    train = datasets.MNIST("", train = True, download = True,
                        transform = transforms.Compose([transforms.ToTensor()]))
    test  = datasets.MNIST("", train = False, download = True,
                        transform = transforms.Compose([transforms.ToTensor()]))

    train_set = torch.utils.data.DataLoader(train, batch_size = 1, shuffle = True)
    test_set  = torch.utils.data.DataLoader(test , batch_size = 1, shuffle = True)
    return train_set, test_set

def save_dataset(threshold):
    train_set, test_set = get_mnist()

    X_train, y_train = np.zeros((len(train_set), 28, 28)), np.zeros(len(train_set), dtype=np.uint8)
    X_test, y_test = np.zeros((len(test_set), 28, 28)), np.zeros(len(test_set), dtype=np.uint8)

    idx = 0
    for data in tqdm(train_set, desc = "Creating Training set: "):
        image = data[0]
        label = data[1]
        X_train[idx] = gray2bin(image, threshold)
        y_train[idx] = np.array(label)
        idx+=1

    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)

    idx = 0
    for data in tqdm(test_set, desc = "Creating Testing set: "):
        image = data[0]
        label = data[1]
        X_test[idx] = gray2bin(image, threshold)
        y_test[idx] = np.array(label)
        idx+=1

    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

    shutil.rmtree('./MNIST', ignore_errors = True)

def arg_parser():
    parser=ArgumentParser()
    parser.add_argument('threshold',type=float, help='threshold to convert grayscale into binary images.')
    return parser

if __name__ == "__main__":
    parser=arg_parser()
    args=parser.parse_args()
    save_dataset(args.threshold)