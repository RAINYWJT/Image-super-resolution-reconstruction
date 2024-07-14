import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

#取图函数
def take_pic(lr_folder,hr_folder):
    #储存高分辨课低分辨的图像：
    lr_images=[]
    hr_images=[]
    #用zip函数进行遍历：
    for lr_file,hr_file in zip(os.listdir(lr_folder),os.listdir(hr_folder)):
        #使用cv2.imread函数读取图像：
        lr_imgs=cv2.imread(os.path.join(lr_folder,lr_file))
        hr_imgs=cv2.imread(os.path.join(hr_folder,lr_file))
        #图像是否为空？
        if lr_imgs is not None and hr_imgs is not None:
            #读取低分辨率宽度和高度：
            height,width,_=lr_imgs.shape
            #如果高度为40*60，就转一下变成60*40
            if height == 40 :
                lr_imgs = cv2.rotate(lr_imgs,cv2.ROTATE_90_CLOCKWISE)
                hr_imgs = cv2.rotate(hr_imgs,cv2.ROTATE_90_CLOCKWISE)
            #添加到列表：
            lr_images.append(lr_imgs)
            hr_images.append(hr_imgs)
    #返回读取的图像列表：
    return lr_images,hr_images

#损失函数
def the_loss(sr , hr):
    #计算方差）
    MSE = torch.mean((sr - hr)**2)
    #psnr峰值信噪比计算：
    psnr=10*torch.log10(MSE)
    return psnr

#训练函数，用SGD优化器，stepLR的学习率调整（num_epoch轮)
def train(model,train_loader,optimizer,criterion,scheduler,device,num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for i in train_loader:
            lr,hr=i
            lr,hr=lr.to(device), hr.to(device)#copy一份
            optimizer.zero_grad()
            #生成超分辨率
            sr=model(lr)
            lost=criterion(sr,hr)#计算损失
            lost.backward()#反向传播
            optimizer.step()#更新
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {lost.item()}")
        scheduler.step()#更新学习率

#验证准备函数，处理高低分辨率的函数
def prepare(lr_images,hr_images):
    #将图像转换为tensor
    transform=transforms.Compose([transforms.ToTensor()])
    #创建训练集返回相应的对象：
    dataset=SR_Dataset(lr_images,hr_images, transform=transform)
    train_loader=DataLoader(dataset,batch_size=20,shuffle=True)#一批20个，乱序（多了泛化性变差）
    return train_loader

#超分辨率处理函数
def SRI(model,image,device):
    #把图像转换为tensor
    transform=transforms.Compose([transforms.ToTensor()])
    image_tensor=transform(image).unsqueeze(0).to(device)#更方便结合数据
    #将图像输入进行超分辨率处理：
    with torch.no_grad():
        sr_tensor=model(image_tensor)
    #还原为numpy
    sr_image=sr_tensor.squeeze(0).cpu().numpy()
    sr_image=np.clip(sr_image,0,1)#截取趋近于0，1的固定值（tensor是一个属于0-1的向量值）
    sr_image = (sr_image * 255).astype("uint8")#bool类型
    sr_image = sr_image.transpose((1, 2, 0))#转置
    return sr_image

#一个神经网络继承自pytorch中nn.Module的类
#这个项目的要求额是40*60扩大8倍
class VDSR(nn.Module):
    #初始化一些东西
    #输入图像通道数：3；,upscale_factor:8降噪处理，八倍；层数：4
    def __init__(self,in_channels=3,upscale_factor=8,num_channels=4):
        super(VDSR,self).__init__()
        #1.神经网络基础设置
        #定义层数：       
        layers=[]
        # 向列表中添加一个卷积层，输入通道数为in_channels, 输出通道数为64，卷积核大小为3x3，stride步幅为1，padding填充为1
        layers.append(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1))
        # 向列表中添加一个ReLU层，inplace=True表示将计算结果直接覆盖到输入数据的存储空间中,加强拟合能力
        # inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
        layers.append(nn.ReLU(inplace=True))
        #接下来添加残差学习块
        for i in range(num_channels):
            layers.append(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1))#卷积核是3
            layers.append(nn.ReLU(inplace=True))
        #因为是8倍数的关系，如果输入通道数是in_channels.那么输出就应该是in_channels*(upscale_factor**2)
        layers.append(nn.Conv2d(64,in_channels*(upscale_factor**2),kernel_size=3,stride=1,padding=1))
        #然后进行聚合重排
        layers.append(nn.PixelShuffle(upscale_factor))
        #2.封装赋值：
        self.My_net = nn.Sequential(*layers)
        #3.初始化权重函数：
        self._initialize()

    #神经网络向前传递函数
    def forward(self,x):
        # 将输入x通过神经网络层self.net进行前向传递运算
        next=self.My_net(x)
        return next

    #神经网络参数初始化函数
    def _initialize(self):
        # 遍历所有网络
        for i in self.modules():
            # 如果是nn.Conv2d
            if isinstance(i,nn.Conv2d):
                #用高斯分布初始化:
                nn.init.kaiming_normal_(i.weight,mode='fan_out',nonlinearity='relu')
                #如果有bias，初始化为0：
                if i.bias is not None:
                    nn.init.constant_(i.bias,0)

#一个继承自Dataset的自定义数据集
class SR_Dataset(Dataset):
    # 初始化方法，接受低分辨率图像lr_images、高分辨率图像hr_images和可选的转换函数transform
    def __init__(self,lr_images,hr_images,transform=None):
        self.lr_images=lr_images # 将输入的低分辨率图像和高分辨率图像赋值给数据集属性
        self.hr_images=hr_images
        self.transform=transform # 将传入的转换函数赋值给数据集属性 

    #返回数据集长度
    def __len__(self):
        return len(self.lr_images) 

    #找到引索的数据对
    def __getitem__(self,index):
        #1.从数据里面读取分别的数据（高分辨率和低分辨率）
        hr=self.hr_images[index]
        lr=self.lr_images[index]
        #2.传入transform函数，对数据进行转换
        if self.transform:
            hr=self.transform(hr)
            lr=self.transform(lr)
        #3.return to data pair：
        return lr,hr

# 主函数：
if __name__=="__main__":
    #1.初始化
    #初始化设备
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #初始化参数
    upscale_factor=8
    num_epochs=100#学习100轮
    input_image_path = "2092.png"
    output_image_path = "re_2092_re_100epoch_4ceng.jpg"
    low_res_folder = "lr"
    high_res_folder = "orig"
    #创建模型
    model=VDSR(3,upscale_factor).to(device)
    #损失函数和优化器
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=1e-3)#优化器（更新参数和学习率）
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs,eta_min=0)#学习率调整(最小学习率和轮次)

    #2.准备数据阶段
    #引入列表（高低分辨率）
    lr_images,hr_images=take_pic(low_res_folder,high_res_folder)
    train_loader=prepare(lr_images,hr_images)
    #训练模型准备
    train(model,train_loader,optimizer,criterion,scheduler,device,num_epochs)
    #读取图像
    input_image=cv2.imread(input_image_path,cv2.IMREAD_COLOR)
    height,width,_=input_image.shape
    flag=0#用于判断是否旋转的bool值
    if height==40:
        flag=1
        input_image=cv2.rotate(input_image,cv2.ROTATE_90_CLOCKWISE)
    input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
    
    #3.处理：
    #使用超分辨率处理
    output_image=SRI(model,input_image,device)
    if flag==1:
        output_image=cv2.rotate(output_image,cv2.ROTATE_90_COUNTERCLOCKWISE)

    #4.保存：
    #保存输出
    output_image=cv2.cvtColor(output_image,cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path,output_image)