# -*- encoding:gbk -*-
# 预训练Resnet101在测试集上得到较高的准确率

import copy
import time

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from torch import randperm
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 设置随机种子，方便实验的复现
torch.manual_seed(42)
batch_size = 8
learning_rate = 0.001
num_epochs = 20

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 运行前清空显存占用
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("此次所用设备为:", device)

# 【加载数据集】

# val_transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# valset = torchvision.datasets.ImageFolder(root='../data/kagglecatsanddogs', transform=val_transform)
# valloader = torch.utils.data.DataLoader(valset, batch_size=5, shuffle=False, num_workers=4)

vis_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

vis_set = torchvision.datasets.ImageFolder(root='../../data/kagglecatsanddogs', transform=vis_transform)
# 加载为dataloader时，不要用numworks，否则Dataloader无法进行迭代。
# vis_loader = torch.utils.data.DataLoader(vis_set, batch_size=batch_size, shuffle=False)
# print(dir(vis_loader))
# print(vis_set[0])
# print(len(vis_set)) # 25000
len_visset = len(vis_set)

# 将数据集进行拆分，首先先要打乱
# randIdx = randperm(len(vis_set)).tolist() # 生成乱序的索引
train_idx, val_idx, test_idx = torch.utils.data.random_split(vis_set, [int(len_visset * 0.5), int(len_visset * 0.3),
                                                                       int(len_visset * 0.2)])  # 切子集，全部切
# print(train_idx)
# print(train_idx.indices) # 是一个乱序的列表
# exit()

# 乱序索引后，取这些索引对应的数据构成一个子集形成训练、验证、测试集合。
train_data = torch.utils.data.Subset(vis_set, train_idx.indices)
val_data = torch.utils.data.Subset(vis_set, val_idx.indices)
test_data = torch.utils.data.Subset(vis_set, test_idx.indices)
# print(train_data[1][1])
# print(train_idx.indices)
# X = val_data[0]
# print(X[0].shape) # torch.Size([3, 224, 224])
# print(X[1]) # 表示的原标签
# plt.imshow(torch.transpose(X[0],0,2)) # 调换两个维度
# print("原本的下标为:", val_data.indices[0])
# plt.show()
# train_data = rand_vis_set.loc[:int(len_visset*0.6)]
# val_data = rand_vis_set.loc[int(len_visset*0.6):int(len_visset*0.8)]
# test_data = rand_vis_set.loc[int(len_visset*0.8):]
# print(len_visset)
# print(len(train_data))
# print(len(val_data))
# print(len(test_data))
# exit()

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

# print(len(train_loader))

'''
# 数据可视化
def imshow(img):
    npimg = img.numpy() # 这个是tensor转为numpy torch.tensor(ndarray) numpy->tensor
    # np_img = np.array(img) # 这个是 imagePIL转为numpy，【但是这边也可以用】
    # print(np_img.shape) # (3, 228, 454)
    # if(npimg == np_img): # 二者并不相等
    #     print("yeah")
    # print(type(npimg)) # <class 'numpy.ndarray'> 这样写会报错
    # print(type(npimg)) # <class 'numpy.ndarray'>
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # 传入的图片是应该是(HWC)，但是npimg是(CHW)，所以得换维度。
    plt.show()

dataiter = iter(vis_loader)
images, labels = dataiter.__next__()
# print(images) # 打印出图片的矩阵数据 == print(torchvision.utils.make_grid(images))
# print(images.shape) # torch.Size([1, 3, 224, 224]) torch.Size([2, 3, 224, 224])
# print(torchvision.utils.make_grid(images).shape) # torch.Size([3, 224, 224]) torch.Size([3, 228, 454]) # 变成一张图像，且长度和高度都有一定的变化
imshow(torchvision.utils.make_grid(images)) # 给图片加上了网格，并且将图片按横排排列成一张图片
# print(type(images)) # <class 'torch.Tensor'>
print(labels)

# for images, labels in vis_loader:
#     imshow(torchvision.utils.make_grid(images))
#     print(labels)
#     exit()
'''


# 【2训练模块】
def train_model(model, criterion, optimizer, scheduler, num_epoches):
    # writer = SummaryWriter("tensorboard_dogscats")
    ep_losses, ep_acces = [], []
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())  # model.state_dict() 获取模型中的所有参数，深度拷贝。
    # best_acc = 0.0
    best_acc = 0.8264
    count = 0
    for epoch in range(num_epoches):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("_" * 10)
        '''
        # 训练和验证交叉传播
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for imgs, labels in tqdm(dataloaders[phase]):
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                # 清空梯度，避免累加了上一批次的梯度
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(imgs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播且仅在训练阶段进行优化
                    if phase == "train":
                        loss.sum().backward()  # 反向传播
                        optimizer.step()

                # 统计loss和准确率
                running_loss += loss.sum().item() * imgs.size(0)
                running_corrects += torch.sum(preds == labels).item()

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / (len(dataloaders[phase]) * batch_size)
            ep_losses.append(epoch_loss)
            ep_acces.append(epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        writer.add_scalar("epoch_loss", {"train":ep_losses[-2], "val":ep_losses[-1]}, global_step=epoch)
        writer.add_scalar("epoch_acc", {"train":ep_acces[-2], "val":ep_acces[-1]}, global_step=epoch)
        writer.close()

        '''
        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_total = 0
        for imgs, labels in tqdm(dataloaders["train"]):
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            # 清空梯度，避免累加了上一批次的梯度
            optimizer.zero_grad()
            outputs = model(imgs)
            # print("输出为: ", outputs)
            # print(outputs.shape) # torch.Size([batchsize, 2])
            # _, preds = torch.max(outputs, 1) # 返回1维度的最大值及其下标
            preds = torch.argmax(outputs, 1) # 返回1维度的最大值的索引下标
            loss = criterion(outputs, labels) # 这边得到的loss是一个scalar
            # print("交叉损失函数值为: ", loss)
            # print(loss.shape)

            # 反向传播且仅在训练阶段进行优化
            # loss.sum().backward()  # 反向传播
            loss.backward()
            optimizer.step()
            # exit()
            # 统计loss和准确率
            with torch.no_grad():
                train_loss += loss.sum().item()
                train_corrects += torch.sum(preds == labels).item()
                train_total += labels.size(0)

        epoch_loss = train_loss / len(dataloaders["train"])
        epoch_acc = float(train_corrects) / float(train_total)
        print("第{}次训练: 损失为:{}, 精确度为:{}".format(epoch, epoch_loss, epoch_acc))
        ep_losses.append(epoch_loss)
        ep_acces.append(epoch_acc)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(dataloaders["val"]):
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                # 清空梯度，避免累加了上一批次的梯度
                optimizer.zero_grad()
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # 统计loss和准确率
                val_loss += loss.item()
                val_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)

            epoch_loss = val_loss / len(dataloaders["val"])
            epoch_acc = float(val_corrects) / float(val_total)
            print("第{}次验证: 损失为:{}, 精确度为:{}".format(epoch, epoch_loss, epoch_acc))
            ep_losses.append(epoch_loss)
            ep_acces.append(epoch_acc)

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # writer.add_scalars("epoch_loss", {"train": ep_losses[-2], "val": ep_losses[-1]}, global_step=epoch)
        # writer.add_scalars("epoch_acc", {"train": ep_acces[-2], "val": ep_acces[-1]}, global_step=epoch)

    # writer.close()

    time_elapsed = time.time() - since
    print(f"training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"best val ACC:{best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "../../BasicNetwork/modelPara/best_resnet101_model_param_1103_1.pth")
    return model


from torchvision.models import resnet101
from torchvision import models

# model = resnet101(pretrained=True) # 这个写法已经不提倡使用
# model = resnet101(weights = models.ResNet101_Weights.IMAGENET1K_V1)
model = resnet101()
# model = nn.DataParallel(model) # 并行计算（无需，只有一个卡）

# 【固定模型参数】
# 固定模型参数的目的是为了在训练过程中只更新需要更新的部分，从而提高模型的训练效率。
# 在PyTorch中，可以通过将需要更新的参数的requires_grad属性设置为False来固定这些参数，
# 而将不需要更新的参数的requires_grad属性设置为True来允许它们进行更新。
# 这样，在反向传播过程中，只有需要更新的参数会被计算梯度并更新，而不需要更新的参数则不会受到影响 。
# 原文链接：https://blog.csdn.net/weixin_43687366/article/details/132654206

for param in model.parameters():
    param.requires_grad = True

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.fc.requires_grad = True
model.to(device)

# print(model.parameters) # 查看网络结构 【迭代地返回 模型所有可学习参数】
# print(model)

criterion = nn.CrossEntropyLoss()
criterion.to(device)
optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=0.9)  # 只优化全连接层的参数

# 【权重衰减】每个5个批次学习率变为原来的lr*0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 将cuda上的参数挪到cpu上
para = torch.load("../../BasicNetwork/modelPara/best_resnet101_model_param_1103_1.pth", map_location=torch.device('cpu'))
model.load_state_dict(para)

#【训练】
best_model = train_model(model, criterion, optimizer, scheduler, num_epochs)

#【测试】

def test_mode(model, criterion):
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    test_total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(dataloaders["test"]):
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, 1)
            loss = criterion(outputs, labels)

            # 统计loss和准确率
            test_loss += loss.sum().item()
            test_corrects += torch.sum(preds == labels).item()
            test_total += labels.size(0) # labels.size(0) = 8 scalar
            # print(type(labels)) # <class 'torch.Tensor'>
            # print(labels.size()) # torch.Size([8])
            # print(labels) # tensor([1, 1, 1, 0, 1, 0, 0, 0])
            # exit()
        test_loss = test_loss / len(dataloaders["test"])
        test_corrects = float(test_corrects) / float(test_total)
        print("测试集上: 损失为:{}, 精确度为:{}".format( test_loss, test_corrects))


test_mode(model, criterion)

