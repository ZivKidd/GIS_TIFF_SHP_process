# -*- coding: utf-8 -*-
import os
from PIL import Image, ImageDraw
import shapefile
import numpy as np


def getFileName(file_dir):
    file_path_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if (file[-3:] == "tif"):
                file_path_list.append(root + '\\' + file)
    return file_path_list


def getAllFileName(folder_path):
    file_list = []
    folder_list = []
    for file_name in os.listdir(folder_path):
        if (os.path.isfile(os.path.join(folder_path, file_name))):
            file_list.append(os.path.join(folder_path, file_name))
        elif (os.path.isdir(os.path.join(folder_path, file_name))):
            folder_list.append(os.path.join(folder_path, file_name))
    file_list.sort()
    return file_list, folder_list


shpfile = r"H:\xzr\process\data2017\Export_Output_2.shp"
folder = r"H:\xzr\process\data2017\tif"
folder_image = r"H:\xzr\process\data2017\image"
folder_mask = r"H:\xzr\process\data2017\mask"
if not os.path.exists(folder_image):
    os.mkdir(folder_image)
if not os.path.exists(folder_mask):
    os.mkdir(folder_mask)

files = []
img_list1 = os.listdir(folder)
img_list = []
for img in img_list1:
    if (img[-4:] == '.tif'):
        img_list.append(img)
img_list.sort(key=lambda x: int(x[:-6]))
img_nums = len(img_list)
for i in range(img_nums):
    img_name = os.path.join(folder, img_list[i])
    files.append(img_name)

sf = shapefile.Reader(shpfile)  # 读取shp文件
shapes = sf.shapes()

for i in range(len(shapes)):

    # if i in [12]:
    #     continue
    #
    # if i>12:
    #     img=Image.open(files[i-1])
    #     print files[i-1]
    # else:
    img = Image.open(files[i])
    print files[i]

    shp = shapes[i]  # 获取shp文件中的每一个形状

    out_image_path = os.path.join(folder_image, str(i) + '.png')
    out_mask_path = os.path.join(folder_mask, str(i) + '.png')

    mask = Image.new("RGB", img.size, (0, 0, 0))

    point = shp.points  # 获取每一个最小外接矩形的四个点
    x_list = [ii[0] for ii in point]
    y_list = [ii[1] for ii in point]

    x_min = min(x_list)
    y_min = min(y_list)
    x_max = max(x_list)
    y_max = max(y_list)

    x_min1 = x_min - (x_max - x_min) * 2
    x_max1 = x_max + (x_max - x_min) * 2
    y_min1 = y_min - (y_max - y_min) * 2
    y_max1 = y_max + (y_max - y_min) * 2

    w = img.size[0] / (x_max1 - x_min1)
    h = img.size[1] / (y_max1 - y_min1)

    shp = np.asarray(shp.points)
    shp[:, 0] -= x_min1
    shp[:, 1] = y_max1 - shp[:, 1]
    shp *= w
    shp = np.asarray(shp, dtype=np.int)
    shp = shp[:-1, :]

    vertice = []
    for i in range(shp.shape[0]):
        vertice.append(tuple(shp[i, :]))

    draw = ImageDraw.Draw(mask)

    draw.polygon(vertice, fill=(255, 255, 255))

    # mask.show()

    mask.save(out_mask_path)
    img.save(out_image_path)
