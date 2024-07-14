from PIL import Image
import os
# 指定存储图片的文件夹路径
folder_path = '/home/rainy/python/graph/ori_pic'
# 指定保存修改后图片的文件夹路径
new_folder_path = '/home/rainy/python/graph/exchange_pic'
# 创建新文件夹
os.makedirs(new_folder_path, exist_ok=True)
# 遍历图片文件夹中所有图片
for file_name in os.listdir(folder_path):
    # 获取图片路径
    file_path = os.path.join(folder_path, file_name)
    # 使用Image模块打开图片
    img = Image.open(file_path)
    # 获取原始图片宽度
    width = img.width
    # 计算16:9高度
    height = int(width * 9 / 16)
    # 进行宽高比例转换
    img = img.resize((width, height))
    # 新图片的路径
    new_file_path = os.path.join(new_folder_path, file_name)
    # 保存新图片
    img.save(new_file_path)