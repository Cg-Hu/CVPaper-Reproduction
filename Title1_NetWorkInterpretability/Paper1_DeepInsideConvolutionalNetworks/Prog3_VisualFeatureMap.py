import PIL.Image
import torch
from torchvision import models
from torch import nn
from torchinfo import summary
from torchvision import transforms
import torchvision
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec ##子图布局模块



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 5
path = "../../data/kagglecatsanddogs/Dog/2.jpg"

#【加载一张图片】
img = PIL.Image.open(path)

trans = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tensor_img = trans(img)
tensor_img = torch.unsqueeze(tensor_img,0)
print(tensor_img.shape)

#【加载模型，并更改模型，并加载模型参数】
resnet101_model = models.resnet101()
num_ftrs = resnet101_model.fc.in_features
resnet101_model.fc = nn.Linear(num_ftrs, 2)

para = torch.load("../../BasicNetwork/modelPara/best_resnet101_model_param_1103_1.pth", map_location=torch.device('cpu'))
resnet101_model.load_state_dict(para)
resnet101_model.to(device)
resnet101_model.eval()
# 查看模型的参数和输出
# summary(resnet101_model,(1,3,224,224))

# print(resnet101_model.layer[1])
# print(resnet101_model.layer1[1])
# print(resnet101_model.layer2)

# 存储中间特征
activate = {}

def get_activation(layer_name):
    def hook(module, input, output):
        # module: model.conv2（要看的哪一层的特征）
        # input :in forward function  [#2]
        # output:is  [#3 self.conv2(out)]
        # clone() 张量复制，但不与原张量共享内存，有梯度属性，但是自身的梯度计算会传给原本的，自身无梯度值。（梯度追溯功能）
        # detach() 张量复制，共享内存，但是无梯度属性，
        # xxx..clone().detach() 就是做简单的复制（另开内存）
        activate[layer_name] = output.clone().detach() # https://blog.csdn.net/dujuancao11/article/details/121563226
        # output is saved  in a list
    return hook
'''
def hook(module, input, output):
    # module: model.conv2（要看的哪一层的特征）
    # input :in forward function  [#2]
    # output:is  [#3 self.conv2(out)]
    feature.append(output.clone().detach())
'''
# register_forward_hook(hook)  最大的作用也就是当训练好某个model，想要展示某一层对最终目标的影响效果。
# 原文链接：https://blog.csdn.net/foneone/article/details/107099060
handle = resnet101_model.layer1.register_forward_hook(get_activation("layer1[0]")) # 获取整个模型layer1层的中间结果
_ = resnet101_model(tensor_img)
# print(resnet101_model.layer1[0])
print(activate["layer1[0]"].shape)
handle.remove()
feature = activate["layer1[0]"]


# 一个画板分成多个部分
gs=GridSpec(5,4) ###指定子图网格大小
fig = plt.figure()
fig.add_subplot(gs[0,0:])
plt.imshow(img)
plt.axis('off')

#展示中间结果
for i in range(4):
    for j in range(4):
        fig.add_subplot(gs[i+1, j])
        plt.imshow(feature[0,i*4+j,:,:], cmap='gray') #因为输出的通道数很多，所以按照通道数来绘制图片
        plt.axis('off')
plt.show()



