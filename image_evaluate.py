import cv2
import numpy as np

# 加载原始图像和处理后的图像
img_org = cv2.imread("/home/rainy/python/graph/orig/2092.png")
img_pro = cv2.imread("/home/rainy/python/graph/2092_re_200epoch_8ceng.jpg")

# 将图像转换为灰度图像
gray_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
gray_pro = cv2.cvtColor(img_pro, cv2.COLOR_BGR2GRAY)

# 计算图像的相关系数
ssim = cv2.matchTemplate(gray_org, gray_pro, cv2.TM_CCORR_NORMED)

# 显示结果
print("SSIM值为:")
print(ssim)

# 获取原始图像的大小
height, width = img_org.shape[:2]

# 定义扩充后的尺寸
new_width, new_height = 320, 480

# 计算水平和竖直方向上的扩充比例
w_ratio, h_ratio = new_width / width, new_height / height

# 保持比例进行扩充
resized_img = cv2.resize(img_org, None, fx=w_ratio, fy=h_ratio, interpolation=cv2.INTER_CUBIC)
resized_img = np.transpose(resized_img, (1, 0, 2))#转置

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# 输出 PSNR 值
print("PSNR值为:")
print(psnr(resized_img,img_pro))