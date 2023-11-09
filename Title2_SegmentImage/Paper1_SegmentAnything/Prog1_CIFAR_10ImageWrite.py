import cv2
import torch
import torchvision
import os
from torch import randperm

train_dataset = torchvision.datasets.CIFAR10(root="../../data", train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root="../../data", train=False, download=True)




def setDir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    # else:
    #     shutil.rmtree(filepath,ignore_errors=True)
        # os.mkdir(filepath)


list_index = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(len(test_dataset)):
    index = test_dataset[i][1]
    name = test_dataset.classes[index]
    image_path = '../../data/cifar_10/test/{}'.format(name)
    image_name = '../../data/cifar_10/test/{}/{:04}.jpg'.format(name, list_index[index])
    setDir(image_path)
    list_index[index] += 1
    test_dataset[0][0].save(image_name)




