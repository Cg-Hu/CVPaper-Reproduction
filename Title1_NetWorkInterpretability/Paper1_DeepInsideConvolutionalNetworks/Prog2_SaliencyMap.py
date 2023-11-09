# -*- encoding:gbk -*-

import torch
from torchvision import models
from torch import nn
from torchvision import transforms
import torchvision
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec ##��ͼ����ģ��

#���豸�������
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 5

#������ģ�ͣ�������ģ�ͣ�������ģ�Ͳ�����
resnet101_model = models.resnet101()
num_ftrs = resnet101_model.fc.in_features
resnet101_model.fc = nn.Linear(num_ftrs, 2)

para = torch.load("../../BasicNetwork/modelPara/best_resnet101_model_param_1103_1.pth", map_location=torch.device('cpu'))
resnet101_model.load_state_dict(para)
resnet101_model.to(device)


# ���������ݡ�
# val_transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# val_set = torchvision.datasets.ImageFolder(root='../../data/kagglecatsanddogs', transform=val_transform)
# val_loader = torch.utils.data.DataLoader(val_set, batch_size=5, shuffle=False)

vis_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
vis_set = torchvision.datasets.ImageFolder(root='../../data/kagglecatsanddogs', transform=vis_transform)
vis_loader = torch.utils.data.DataLoader(vis_set, batch_size=5, shuffle=False)


vis_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

vis_set = torchvision.datasets.ImageFolder(root='../../data/kagglecatsanddogs', transform=vis_transform)
# ����Ϊdataloaderʱ����Ҫ��numworks������Dataloader�޷����е�����
# vis_loader = torch.utils.data.DataLoader(vis_set, batch_size=batch_size, shuffle=False)
# print(dir(vis_loader))
# print(vis_set[0])
# print(len(vis_set)) # 25000
len_visset = len(vis_set)

# �����ݼ����в�֣�������Ҫ����
# randIdx = randperm(len(vis_set)).tolist() # �������������
train_idx, val_idx, test_idx = torch.utils.data.random_split(vis_set, [int(len_visset * 0.5), int(len_visset * 0.3),
                                                                       int(len_visset * 0.2)])  # ���Ӽ���ȫ����
train_data = torch.utils.data.Subset(vis_set, train_idx.indices)
val_data = torch.utils.data.Subset(vis_set, val_idx.indices)
test_data = torch.utils.data.Subset(vis_set, test_idx.indices)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# ����ͼ
resnet101_model.eval() # ����ͼ����ô����Ĳ�������䣬�̶�ס��
for para in resnet101_model.parameters():
    para.requires_grad = False

val_data_iter = iter(val_loader)
x, y = val_data_iter.__next__()
x = x.to(device)
x.requires_grad_() # xͼ����Ҫ�ݶ�
y = y.to(device)
saliency = None

outputs = resnet101_model.forward(x)
# print(outputs)
outputs = outputs.gather(1, y.view(-1, 1)).squeeze() # �õ���ȷ����
# ���򴫲�����õ�����ȡ����δ��softmax��ֵ���ڸ�������ͼ�����صĵ�����
outputs.backward(torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float32)) # ֻ������ȷ���ಿ�ֵ�loss


# print(x.grad.shape) # [5,3,224,224]
# exit()


saliency = abs(x.grad.data) # ����X���ݶȾ���ֵ��С
saliency, _ = torch.max(saliency, dim=1)  # 3��ͨ����ֻȡ���ֵ���ݶȣ������ݶ�
print(saliency.shape) # torch.Size([5, 224, 224])
# exit()
# saliency.squeeze() # ����һ��ά
# print(saliency.shape) # torch.Size([5, 224, 224])
# exit()

gs=GridSpec(3,5) ###ָ����ͼ�����С

def imshow(img, fig):
    npimg = img.numpy()
    fig.add_subplot(gs[0,0:])
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

vis_data_iter = iter(val_loader)
images, labels = vis_data_iter.__next__()
fig = plt.figure()
imshow(torchvision.utils.make_grid(images), fig)

# norm = transforms.Normalize([0.485], [0.229])
# norm(saliency)
saliency = saliency.numpy()


for i in range(5):
    fig.add_subplot(gs[1,i:i+1])
    plt.imshow(saliency[i], cmap=plt.cm.hot)
    plt.axis('off')
    fig.add_subplot(gs[2,i:i+1])
    plt.imshow(saliency[i])
    plt.axis('off')
    # plt.gcf().set_size_inches(12, 5)

plt.show()



