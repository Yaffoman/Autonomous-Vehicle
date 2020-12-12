from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
#
# left = []
# right = []
#
# image_list = []
# for filename in glob.glob('Dataset/00033/*.ppm'):
#     im=Image.open(filename)
#     im = im.resize((50, 50))
#     im = ImageOps.grayscale(im)
#     right.append(np.asarray(im))
#
# for filename in glob.glob('Dataset/00034/*.ppm'):
#     im = Image.open(filename)
#     im = im.resize((50, 50))
#     im = ImageOps.grayscale(im)
#     left.append(np.asarray(im))
#
# plt.imshow(left[20], cmap='gray')
# plt.show()
#
# X_train = np.concatenate((left, right))
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
# y_train = np.concatenate((np.zeros(len(left)), np.ones(len(right))))
# print("Total Data:\n", X_train.shape, "XShape\n", y_train.shape, "yShape")


class ArrowDataset(Dataset):

    # You can read the train_list.txt and test_list.txt files here.
    def __init__(self, device='cpu', val=0, test=0):
        self.data = []
        self.labels = []
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.left = []
        self.right = []

        for filename in glob.glob('Dataset/00034/*.ppm'):
            im = Image.open(filename)
            im = im.resize((50, 50))
            im = self.transform(im)
            self.left.append(im)
            self.labels.append(1)  # Left

        for filename in glob.glob('Dataset/00033/*.ppm'):
            im = Image.open(filename)
            im = im.resize((50, 50))
            im = self.transform(im)
            self.right.append(im)
            self.labels.append(0)  # Right


        if val > 0:
            self.data = self.left[:val] + self.right[-val:]
            self.labels = self.labels[:val] + self.labels[-val:]
        elif test > 0:
            self.data = self.left[-test:] + self.right[:test]
            self.labels = self.labels[len(self.left)-test:len(self.left)+test]
        else:
            self.data = self.left[50:-50] + self.right[50:-50]
            self.labels = self.labels[50:len(self.left)-50] + self.labels[len(self.left)+50:-50]
        self.labels = torch.LongTensor(self.labels)

        print("Total Data:\n", len(self.data), "XShape\n", len(self.labels), "yShape")

    def __len__(self):
        return len(self.data)

    # Reshape image to (224,224).
    # Try normalizing with imagenet mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] or
    # any standard normalization
    # You can other image transformation techniques too
    def __getitem__(self, item):
        # SHOULD LOAD IMAGES HERE!!!
        return self.data[item].to(self.device), self.labels[item].to(self.device)


class TestingDataset(Dataset):

    # You can read the train_list.txt and test_list.txt files here.
    def __init__(self, device='cpu'):
        self.data = []
        self.labels = []
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        for filename in glob.glob('Dataset/Test/*'):
            im = Image.open(filename).convert('RGB')
            im = im.resize((50, 50))
            im = self.transform(im)
            self.data.append(im)
            self.labels.append(1)  # Left

        self.labels = torch.LongTensor(self.labels)

        print("Total Data:\n", len(self.data), "XShape\n", len(self.labels), "yShape")

    def __len__(self):
        return len(self.data)

    # Reshape image to (224,224).
    # Try normalizing with imagenet mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] or
    # any standard normalization
    # You can other image transformation techniques too
    def __getitem__(self, item):
        # SHOULD LOAD IMAGES HERE!!!
        return self.data[item].to(self.device), self.labels[item].to(self.device)
