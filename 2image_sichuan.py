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

def getFilesForGivenIndex(index,file_list):
    result=[]
    for f in file_list:
        name=os.path.split(f)[1]
        name=name.split('-')[0]
        name=int(name)
        if(name==index):
            result.append(f)
    return result

shpfile = r"D:\deepleearning\sampleshp\titian\titian.shp"
folder = r"D:\deepleearning\sampleshp\titian\tif_clip"
folder_image = r"D:\deepleearning\sampleshp\titian\image"
folder_mask = r"D:\deepleearning\sampleshp\titian\mask"
# 滑坡
# color=(255, 255, 255)
# #崩塌
#color=(0, 255, 0)
# #泥石流
#color=(0, 0, 255)
# 采石场
color=(0, 0, 0)

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
# img_list.sort(key=lambda x: int(x[:-6]))
img_nums = len(img_list)
for i in range(img_nums):
    img_name = os.path.join(folder, img_list[i])
    files.append(img_name)

sf = shapefile.Reader(shpfile)  # 读取shp文件
shapes = sf.shapes()

n=0
for i in range(len(shapes)):

    # if i <n:
    #     continue
    #
    # if i>12:
    #     img=Image.open(files[i-1])
    #     print files[i-1]
    # else:
    files1=getFilesForGivenIndex(i,files)
    print i
    for j in range(len(files1)):
        n+=1
        img = Image.open(files1[j])
        print files1[j]

        shp = shapes[i]  # 获取shp文件中的每一个形状

        out_image_path = os.path.join(folder_image, str(n) + '.png')
        out_mask_path = os.path.join(folder_mask, str(n) + '.png')

        mask = Image.new("RGB", img.size, (0, 0, 0))

        point = shp.points  # 获取每一个最小外接矩形的四个点
        x_list = [ii[0] for ii in point]
        y_list = [ii[1] for ii in point]

        x_min = min(x_list)
        y_min = min(y_list)
        x_max = max(x_list)
        y_max = max(y_list)

        # x_min1 = x_min - (x_max - x_min) / 2
        # x_max1 = x_max + (x_max - x_min) / 2
        # y_min1 = y_min - (y_max - y_min) / 2
        # y_max1 = y_max + (y_max - y_min) / 2
        #
        # x_min1 = x_min - (x_max - x_min) * 2
        # x_max1 = x_max + (x_max - x_min) * 2
        # y_min1 = y_min - (y_max - y_min) * 2
        # y_max1 = y_max + (y_max - y_min) * 2

        x_min1 = x_min - (x_max - x_min)
        x_max1 = x_max + (x_max - x_min)
        y_min1 = y_min - (y_max - y_min)
        y_max1 = y_max + (y_max - y_min)

        w = img.size[0] / (x_max1 - x_min1)
        h = img.size[1] / (y_max1 - y_min1)

        shp = np.asarray(shp.points)
        shp[:, 0] -= x_min1
        shp[:, 1] = y_max1 - shp[:, 1]
        shp[:, 0] *= w
        shp[:, 1] *= h

        shp = np.asarray(shp, dtype=np.int)
        shp = shp[:-1, :]

        vertice = []
        for k in range(shp.shape[0]):
            vertice.append(tuple(shp[k, :]))

        draw = ImageDraw.Draw(mask)

        #滑坡
        draw.polygon(vertice, fill=color)


        # mask.show()

        mask.save(out_mask_path)
        img.save(out_image_path)
