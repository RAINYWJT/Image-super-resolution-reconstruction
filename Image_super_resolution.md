# 图像超分辨率处理：
## 一.总述
### 1.选题描述：
    图像超分辨率重建（Image Super-Resolution，ISR）是指将低分辨率（LR）图像转换为⾼分辨率（HR）图像的过程。这个问题在许多应⽤中都很重要，例如医学图像分析、视频监控、卫星图像处理等领域。
    传统的ISR⽅法通常使⽤插值⽅法或基于图像统计模型的⽅法。插值⽅法通常是将每个LR像素的值通过⼀些插值技术扩展到HR像素的值，例如双线性插值、三次样条插值等等。这些⽅法在⼀些简单的场景下效果不错，但对于复杂的场景来说，插值⽅法并不能提供令⼈满意的结果。近年来，深度学习⽅法在ISR领域取得了巨⼤的成功。这些⽅法通常使⽤卷积神经⽹络（Convolutional Neural Networks，CNN）进⾏训练，以将LR图像映射到HR图像。ISR技术的发展使得在图像处理中实现更⾼的分辨率成为可能，为⼈们提供了更加清晰的图像和更好的视觉体验。

    任务介绍
        1. 熟悉并学习使⽤基本的opencv库、卷积操作和反卷积操作：通常⼀张图⽚有3个维度组成：C，H，W。其中彩⾊图像的C通常为3，灰度图像为1。H和W分别为图⽚的⻓宽。数据集的描述：lr⽂件夹中是所有低分辨率图⽚(60 * 40或者40 * 60)，orig⽂件夹中是对应的⾼分辨率图⽚（480 * 320或者320 * 480）
        2. 了解图像超分辨率的评价指标：Peak Signal-to-Noise Ratio(PSNR)和structural similarityindex(SSIM)，并编写代码实现。设计损失函数
        3. 编写代码使⽤Bicubic interpolation实现超分辨率，并报告⽣成的图像与原始图像的PSNR和SSIM
        4. 使⽤深度神经⽹络实现超分辨率重建，并报告与Bicubic interpolation结果的对⽐。

### 2.设计方案：
    首先：opencv库里面有很多处理图像的函数，rotate，imread...
        还有读取文件的函数：os.path.join()是必须的
        torch.nn，torch.optim，torchvision.transforms，torch.utils.dat这些是pytorch的一些处理库，有用，拿上
    接着要实现一些功能：
    tips
```python
我们在定义自已的网络的时候，需要继承nn.Module类，并重新实现构造函数__init__构造函数和forward这两个方法。但有一些注意技巧：

（1）一般把网络中具有可学习参数的层（如全连接层、卷积层等）放在构造函数__init__()中，当然我也可以吧不具有参数的层也放在里面；

（2）一般把不具有可学习参数的层(如ReLU、dropout、BatchNormanation层)可放在构造函数中，也可不放在构造函数中，如果不放在构造函数__init__里面，则在forward方法里面可以使用nn.functional来代替
    
（3）forward方法是必须要重写的，它是实现模型的功能，实现各个层之间的连接关系的核心。
```
        所以可以得到我们需要实现的基本功能：
```python
#取图函数
def take_pic():
    pass

#损失函数
def the_loss():
    pass

#训练函数
def train():
    pass

#验证准备函数
def prepare():
    pass

#超分辨率处理函数
def SRI():
    pass

#一个神经网络继承自pytorch中nn.Module的类
class VDSR(nn.Module):
    #初始化一些东西
    def __init__():
        pass
    #神经网络向前传递函数
    def forward():
        pass
    #神经网络参数初始化函数
    def initialize():
        pass

#一个继承自Dataset的自定义数据集
class SR_Dataset(Dataset):
    #初始化方法
    def __init__():
        pass
    #返回数据集长度
    def __len__(self):
        pass
    #找到引索的数据对
    def __getitem__():
        pass

# 主函数：
if __name__=="__main__":
    #1.初始化
    #初始化设备
    #初始化参数
    #创建模型
    #损失函数和优化器
    
    #2.准备数据阶段
    #引入列表（高低分辨率）
    #训练模型
    #读取图像

    #3.处理：
    #使用超分辨率处理

    #4.保存：
    #保存输出
```
        ok,以上就是我们需要实现的功能和这个处理器的基本内核，接下来进行代码实现
### 3.功能划分与描述：
        class类：
        (1).VDSR(nn.Module):
            参考：https://zhuanlan.zhihu.com/p/549975513
            参考2：https://zhuanlan.zhihu.com/p/203405689
            参考3：https://zhuanlan.zhihu.com/p/613392544
            参考4：https://zhuanlan.zhihu.com/p/557253923
            参考5：https://zhuanlan.zhihu.com/p/455442102
            参考6：https://zhuanlan.zhihu.com/p/562932795
            参考7：https://zh.d2l.ai/chapter_convolutional-modern/resnet.html
            参考8：https://zhuanlan.zhihu.com/p/428448728
            参考9：https://blog.csdn.net/weixin_42667163/article/details/125392574?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168181777116800227467652%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168181777116800227467652&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-125392574-null-null.142^v84^koosearch_v1,239^v2^insert_chatgpt&utm_term=nn.init.kaiming_normal_%28m.weight%2C%20mode%3Dfan_out%2C%20nonlinearity%3Drelu%29&spm=1018.2226.3001.4187
```python
class VDSR(nn.Module):
    def __init__(self, in_channels) -> None:
        super(VDSR, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=64, kernel_size=3,stride=1, padding=1),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64,in_channels,kernel_size=3, stride=1,padding=1)
        )

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = x + y 
        return y 
```
    以下是我的实现：
```python
#一个神经网络继承自pytorch中nn.Module的类
#这个项目的要求额是40*60扩大8倍
class VDSR(nn.Module):
    #初始化一些东西
    #输入图像通道数：3；,upscale_factor:8降噪处理，八倍；残差：4
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
    def _initialize():
        # 遍历所有网络
        for i in self.modules():
            # 如果是nn.Conv2d
            if isinstance(i,nn.Conv2d):
                #用高斯分布初始化“
                nn.init.kaiming_normal_(i.weight,mode='fan_out',nonlinearity='relu')
                #如果有bias，初始化为0：
                if i.bias is not None:
                    nn.init.constant_(i.bias,0)
```
        (2).SRDataset(Dataset):
            参考：https://zhuanlan.zhihu.com/p/105507334
```python
    class CustomDataset(data.Dataset):#需要继承data.Dataset
    def __init__(self):
        # TODO
        # 1. Initialize file path or list of file names.
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0
```
    以下是我的实现：
```python
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
```
        这个class就是根据上面的结构完成的

        （3）其他函数：
    1.取图函数：
    参考10：https://blog.csdn.net/kk185800961/article/details/79307736?ops_request_misc=&request_id=&biz_id=102&utm_term=load_images_from_folder(lr_fol&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-79307736.142^v84^koosearch_v1,239^v2^insert_chatgpt&spm=1018.2226.3001.4187
```python
#取图函数
def take_pic():
    #储存高分辨课低分辨的图像：
    lr_images=[]
    hr_images=[]
    #用zip函数进行遍历：
    for lr_file,hr_file in zip(os.listdir(lr_folder),os.listdir(hr_folder))
        #使用cv2.imread函数读取图像：
        lr_imgs=cv2.imread(os.path.join(lr_folder,lr_file))
        hr_imgs=cv2,imread(os.path.join(hr_folder,lr_file))
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
```

    2.损失函数：
    参考：https://blog.csdn.net/shentu7/article/details/105947170/?ops_request_misc=&request_id=&biz_id=102&utm_term=PSNR%EF%BC%88%E5%B3%B0%E5%80%BC%E4%BF%A1%E5%99%AA%E6%AF%94%EF%BC%89%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-5-105947170.142^v84^koosearch_v1,239^v2^insert_chatgpt&spm=1018.2226.3001.4187
```python
#损失函数
def the_loss(sr , hr):
    #计算方差）
    MSE = torch.mean((sr - hr)**2)
    #psnr峰值信噪比计算：
    psnr=10*torch.log10(MSE)
    return psnr
```

    3.训练函数：
    参考：https://blog.csdn.net/tcn760/article/details/123965374?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168182516416800182152925%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168182516416800182152925&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-123965374-null-null.142^v84^koosearch_v1,239^v2^insert_chatgpt&utm_term=SGD%E4%BC%98%E5%8C%96%E5%99%A8&spm=1018.2226.3001.4187
    参考2：https://blog.csdn.net/maly_Sunshine/article/details/123225799?ops_request_misc=&request_id=&biz_id=102&utm_term=&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-123225799.142^v84^koosearch_v1,239^v2^insert_chatgpt&spm=1018.2226.3001.4187#%20%E8%AE%AD%E7%BB%83%E5%87%BD%E6%95%B0%EF%BC%8C%E9%87%87%E7%94%A8SGD%E4%BC%98%E5%8C%96%E5%99%A8%EF%BC%8CStepLR%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4%E6%96%B9%E5%BC%8F&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-123225799.142^v84^koosearch_v1,239^v2^insert_chatgpt
```python
#训练函数，用SGD优化器，stepLR的学习率调整
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
```

    4.验证准备函数：
    参考：https://blog.csdn.net/weixin_42535423/article/details/122106480?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%B0%86%E5%9B%BE%E5%83%8F%E8%BD%AC%E5%8C%96%E4%B8%BAtensor%E7%B1%BB%E5%9E%8B&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-122106480.142^v84^koosearch_v1,239^v2^insert_chatgpt&spm=1018.2226.3001.4187
```python
#验证准备函数，处理高低分辨率的函数
def prepare(lr_images,hr_images):
    #将图像转换为tensor
    transform=transforms.Compose([transforms.ToTensor()])
    #创建训练集返回相应的对象：
    dataset=SR_Dataset(lr_images,hr_images, transform=transform)
    train_loader=DataLoader(dataset,batch_size=20,shuffle=True)
    return train_loader
```

    5.超分辨率处理函数：
    参考：https://blog.csdn.net/leo0308/article/details/124578677?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168182826916800188558387%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168182826916800188558387&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-124578677-null-null.142^v84^koosearch_v1,239^v2^insert_chatgpt&utm_term=torch.no_grad%28%29%3A&spm=1018.2226.3001.4187
    参考2：https://blog.csdn.net/qq_39180879/article/details/108754912?ops_request_misc=&request_id=&biz_id=102&utm_term=tensor.squeeze(0).cpu().numpy(&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-108754912.142^v84^koosearch_v1,239^v2^insert_chatgpt&spm=1018.2226.3001.4187
```python
#超分辨率处理函数
def SRI(model,image,device):
    #把图像转换为tensor
    transform=transforms.Compose([transforms.ToTensor()])
    image_tensor=transform(image),upsqueeze(0).to(device)#更方便结合数据
    #将图像输入进行超分辨率处理：
    with torch.no_grad():
        sr_tensor=model(image_tensor)
    #还原为numpy
    sr_image=sr_tensor.squeeze(0).cpu().numpy()
    sr_image=np.clip(sr_image,0,1)#截取趋近于0，1的固定值
    sr_image = (sr_image * 255).astype("uint8")#bool类型
    sr_image = sr_image.transpose((1, 2, 0))#转置
    return sr_image
```

        (4).主程序：
    参考：https://blog.csdn.net/baidu_41879652/article/details/118307330?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168182916816800192271722%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168182916816800192271722&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-118307330-null-null.142^v84^koosearch_v1,239^v2^insert_chatgpt&utm_term=torch.device&spm=1018.2226.3001.4187
    参考2：https://blog.csdn.net/weixin_36670529/article/details/104349560?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-1-104349560-blog-118307330.235^v29^pc_relevant_default_base3&spm=1001.2101.3001.4242.2&utm_relevant_index=4
    参考3：https://blog.csdn.net/lj2048/article/details/114889359?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168182975916800213014255%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168182975916800213014255&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-114889359-null-null.142^v84^koosearch_v1,239^v2^insert_chatgpt&utm_term=optim.Adam&spm=1018.2226.3001.4187
    参考4：https://blog.csdn.net/zisuina_2/article/details/103250274?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168183001216800184126526%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168183001216800184126526&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-103250274-null-null.142^v84^koosearch_v1,239^v2^insert_chatgpt&utm_term=optim.lr_scheduler.CosineAnnealingLR&spm=1018.2226.3001.4187
    参考5：https://blog.csdn.net/weixin_44015965/article/details/109547129?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168183157016800217234117%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168183157016800217234117&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-109547129-null-null.142^v84^koosearch_v1,239^v2^insert_chatgpt&utm_term=cv2.imread&spm=1018.2226.3001.4187
    参考6：https://blog.csdn.net/weixin_42730667/article/details/102299280?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168183217116800225510259%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168183217116800225510259&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-102299280-null-null.142^v84^koosearch_v1,239^v2^insert_chatgpt&utm_term=cvtColor&spm=1018.2226.3001.4187
    参考7：https://blog.csdn.net/weixin_52527544/article/details/128008221?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168188799816800192243678%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168188799816800192243678&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-128008221-null-null.142^v84^koosearch_v1,239^v2^insert_chatgpt&utm_term=BGR%20RGB&spm=1018.2226.3001.4187
```python
# 主函数：
if __name__=="__main__":
    #1.初始化
    #初始化设备
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #初始化参数
    upscale_factor=8
    num_epochs=100#学习100轮
    input_image_path = "2092.png"
    output_image_path = "1.jpg"
    low_res_folder = "lr"
    high_res_folder = "orig"
    #创建模型
    model=VDSR(3,upscale_factor).to(device)
    #损失函数和优化器
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=1e-3)#优化器
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
        input_image=cv.rotate(input_image,cv.ROTATE_90_CLOCKWISE)
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
```
### 4.实现效果：
先谈一下问题，在训练之后我找了一个双三次插值的测试图来对比，结果发现只有28.23左右

<img src="/home/rainy/python/graph/2092_re_200epoch_4ceng.jpg" width="480" height="320"/>

上面这个图是4层卷积，200轮学习的效果

<img src="/home/rainy/python/graph/2092_re_100epoch_4ceng.jpg" width="480" height="320"/>

上面这个图是4层卷积，100轮学习的效果

<img src="/home/rainy/python/graph/2092_re_100epoch_8ceng.jpg" width="480" height="320"/>

上面这个图是8层卷积，100轮学习的效果

以下是双三次插值和生成图的对比

而且生成图和插值的图差距仅仅拉开了0.12，明显是不对的，所以大概研究了一下，发现是代码写的有问题放大倍数过大，到了8*8
更改后用插值和生成图比，得到了一些值：

以下是100轮学习的结果对比

| 图片名 | SSIM | PSNR |
| ------------- | ------------ | ----------- |
| 插值 | 对照组   | 对照组  |
| 2092_re_100epoch_3ceng.jpg | [[0.999628]]   |  38.30601243085256  |
| 2092_re_100epoch_4ceng.jpg | [[0.9996214]]   |  38.32772640581786  |
| 2092_re_100epoch_5ceng.jpg | [[0.99963117]]   |  38.26197583696789  |
| 2092_re_100epoch_6ceng.jpg | [[0.9995541]]   |  37.64376798932218  |
| 2092_re_100epoch_8ceng.jpg | [[0.9995236]]   |  37.46091323289682  |
| 2092_re_100epoch_10ceng.jpg | [[0.9994777]]   |  36.943542588671455  |

不难看出，在100轮学习的前提下，选择四层卷积神经网络所得到的效果是最好的，层数过多会导致过拟合

以下是200轮学习的结果对比

| 图片名 | SSIM | PSNR |
| ------------- | ------------ | ----------- | 
| 插值 | 对照组   | 对照组  |
| 2092_re_200epoch_4ceng.jpg | [[0.99955094]]   |  37.887482963583466  |
| 2092_re_200epoch_6ceng.jpg | [[0.9995504]]   |  37.79841562698215  |
| 2092_re_200epoch_8ceng.jpg | [[0.9995681]]  |  37.895664977358756  |

可以看到，两百轮学习出现了一个很怪异的现象，居然在第8轮的时候上升了，我们再训练一个10层的数据来看一下

| 图片名 | SSIM | PSNR |
| ------------- | ------------ | ----------- |
| 插值 | 对照组   | 对照组  |
| 2092_re_200epoch_10ceng.jpg | [[0.999537]]   |  37.70682870809226  |

经过测试，发现是个意外，所以可以不用管这个事
那基本经过测定，100轮训练下，4层卷积网络能达到最好的训练效果


以下是插值图和高清图的对比：

| 图片名 | SSIM | PSNR |
| ------------- | ------------ | ----------- |
| 插值 | 对照组   | 对照组  |
| Bicubic_interpolation_pic.jpg | [[0.9950384]]   |  32.73714804932942  |

抛开这个结果不谈，比上面低了很多，然后我们拿生成的学习图片和高清图进行对比

| 图片名 | SSIM | PSNR |
| ------------- | ------------ | ----------- |
| 插值 | 对照组   | 对照组  |
| 2092_re_100epoch_3ceng.jpg | [[0.99498755]]  |  32.56392661645516  |
| 2092_re_100epoch_4ceng.jpg | [[0.9950015]]   |  32.57234757777346 |
| 2092_re_100epoch_5ceng.jpg | [[0.99495196]]   |  32.53920910111842 |
| 2092_re_100epoch_6ceng.jpg | [[0.9948728]]   |  32.463371884044754 |
| 2092_re_100epoch_8ceng.jpg | [[0.9948623]]   |  32.48282960023227 |
| 2092_re_100epoch_10ceng.jpg | [[0.9946912]]   |  32.285855833209496 |

可见，100轮时学习多了反而下降，而且并没有达到预期的目的：
然后对比200轮

| 图片名 | SSIM | PSNR |
| ------------- | ------------ | ----------- | 
| 插值 | 对照组   | 对照组  |
| 2092_re_200epoch_4ceng.jpg | [[0.99505067]]   |  32.662955584225614  |
| 2092_re_200epoch_6ceng.jpg | [[0.9950266]]   |  32.628032405634904  |
| 2092_re_200epoch_8ceng.jpg | [[0.9949381]]  |  32.54339846146174  |
| 2092_re_200epoch_10ceng.jpg | [[0.99487674]]  |  32.50458889131466  |

然后对比300轮：

| 图片名 | SSIM | PSNR |
| ------------- | ------------ | ----------- | 
| 插值 | 对照组   | 对照组  |
| 2092_re_300epoch_4ceng.jpg | [[0.9950848]]   |  32.68778859823095 |

然后对比400轮：

| 图片名 | SSIM | PSNR |
| ------------- | ------------ | ----------- | 
| 插值 | 对照组   | 对照组  |
| 2092_re_400epoch_4ceng.jpg | [[0.995099]]   |  32.69349855258482 |

经过分析，发现所得到的图像就是比双三次插值垃圾，呃啊，目前估计问题出在图像过少，放大倍数过多，导致跟双三次插值所出的效果差不多

以下展览效果最好的图,及对比

<img src="/home/rainy/python/graph/2092_re_400epoch_4ceng.jpg" width="480" height="320"/>

<img src="/home/rainy/python/graph/Bicubic_interpolation_pic.jpg" width="480" height="320"/>

