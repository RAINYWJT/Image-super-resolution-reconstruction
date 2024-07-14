# 导入Pillow库
from PIL import Image
import os
# 定义源文件夹和目标文件夹的路径
source_folder = "/home/rainy/python/graph/exchange_pic"
target_folder = "/home/rainy/python/graph/exchange_pic_two"

# 定义目标图片大小为3MB，单位为字节
target_size = 3 * 1024 * 1024

# 遍历源文件夹中的所有png
for file in os.listdir(source_folder):
    if file.endswith(".png"):
        # 打开图片文件
        image = Image.open(os.path.join(source_folder, file))
        # 获取图片的宽度和高度
        width, height = image.size
        # 计算图片的压缩比例，假设图片是RGB格式，每个像素占3字节
        ratio = (width * height * 3 / target_size) ** 0.5
        # 如果压缩比例大于1，说明图片需要缩小
        if ratio > 1:
            # 按照压缩比例缩小图片的宽度和高度
            new_width = int(width / ratio)
            new_height = int(height / ratio)
            # 重新调整图片的大小
            image = image.resize((new_width, new_height))
        # 保存图片到目标文件夹，使用质量参数为85
        image.save(os.path.join(target_folder, file), quality=85)
