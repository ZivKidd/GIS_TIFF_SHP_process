import gdal
import numpy as np
import os
from PIL import Image, ImageDraw
from shutil import copyfile

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


cell_size=7.6445863e-06
top=27.5632738994
left=102.928037623
l=(103.3782-left)/cell_size
r=(103.4148-left)/cell_size
t=(top-27.466)/cell_size
d=(top-27.3722)/cell_size

file_list,_=getAllFileName(r"H:\xzr\duxiang\clip_1024")
folder=r'H:\xzr\duxiang\cloud\\'
for f in file_list:
    if(f[-4:]!='.tif'):
        continue
    name=os.path.split(f)[1][:-4]
    name=name.split('-')
    x=int(name[0])
    y=int(name[1])

    if(x<r and x>l):
        if(y>t and y<d):
            new_name=folder+os.path.split(f)[1]
            copyfile(f,new_name)
            print(f)

