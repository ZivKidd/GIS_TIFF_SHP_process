import gdal
import numpy as np
import os
from PIL import Image, ImageDraw
import datetime

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


# raster = np.zeros((1000,2000), dtype=np.int)
# raster[200:300,1000:1800]=255
#
# # 读取要切的原图
# in_ds = gdal.Open(r"H:\xzr\duxiang\压缩文件\201912_GF2.img")
# in_band1 = in_ds.GetRasterBand(1)
#
# dst_ds = gdal.GetDriverByName('GTiff').Create("hello.tif", 2000, 1000, 1, in_band1.DataType)
# dst_ds.SetGeoTransform(in_ds.GetGeoTransform())
# dst_ds.SetProjection(in_ds.GetProjection())
# dst_ds.GetRasterBand(1).WriteArray(raster)
# # Once we're done, close properly the dataset
# dst_ds.FlushCache()



raster = np.zeros((78234,147474), dtype=np.int)


file_list,_=getAllFileName(r"D:\semantic-segmentation-for-Geographical-survey\Val1")
num=0
for f in file_list:
    num += 1
    if(num%1000==0):
        print(datetime.datetime.now())
        print(num,'/25731')

    name=os.path.split(f)[1][:-4]
    name=name.split('-')
    x=int(name[0])
    y=int(name[1])

    image=Image.open(f)
    image=image.resize((1024,1024))
    image = np.array(image)  # image类 转 numpy
    image = image[:, :, 0]  # 第1通道

    if(np.max(image)==0):
        continue

    has_value=np.where(image>0)
    for v in range(has_value[0].shape[0]):
        raster[has_value[0][v]+y,has_value[1][v]+x]=255

in_ds = gdal.Open(r"H:\xzr\duxiang\压缩文件\201912_GF2.img")
in_band1 = in_ds.GetRasterBand(1)

dst_ds = gdal.GetDriverByName('GTiff').Create("hello4.tif", 147474, 78234, 1, in_band1.DataType)
dst_ds.SetGeoTransform(in_ds.GetGeoTransform())
dst_ds.SetProjection(in_ds.GetProjection())
dst_ds.GetRasterBand(1).WriteArray(raster)
# Once we're done, close properly the dataset
dst_ds.FlushCache()
