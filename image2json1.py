import gdal
import numpy as np
import os
from PIL import Image, ImageDraw
import datetime
import cv2
import json

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

in_ds = gdal.Open(r"H:\xzr\duxiang\压缩文件\201912_GF2.img")

gt = in_ds.GetGeoTransform()
x0 = gt[0]
x_delta = gt[1]
y0 = gt[3]
y_delta = gt[-1]


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


result=[]
num=1

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
img = cv2.morphologyEx(raster, cv2.MORPH_CLOSE, kernel)
img = cv2.morphologyEx(raster, cv2.MORPH_OPEN, kernel)
print(0)
#     img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
#
#     image, contours, hierarch = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if (len(contours) == 0):
#         continue
#
#     list1 = []
#     for i in range(len(contours)):
#         area = cv2.contourArea(contours[i])
#         if (area < 1600):
#             continue
#         x, y, w, h = cv2.boundingRect(contours[i])
#         print(x, y, w, h)
#
#         xl = x1 + x
#         yt = y1 + y
#         xr = xl + w
#         yd = yt + h
#
#         xlc = xl * x_delta + x0
#         xrc = xr * x_delta + x0
#         ytc = yt * y_delta + y0
#         ydc = yd * y_delta + y0
#
#         rect={}
#
#         rect['type']= "Feature"
#         rect['properties']={'index':num}
#
#         points=[[[xlc,ytc],[xlc,ydc],[xrc,ydc],[xrc,ytc],[xlc,ytc]]]
#
#         rect["geometry"]={}
#         rect["geometry"]["type"]='Polygon'
#         rect["geometry"]["coordinates"]=points
#
#
#
#         num+=1
#         result.append(rect)
#
# result1={}
# result1['type']='FeatureCollection'
# result1['name']='duxiangV1'
# result1["crs"]= { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } }
# result1['features']=result
#
# with open('duxiangV1.geojson', 'w') as f:
#     json.dump(result1, f)
# print(0)

        # cv2.namedWindow("Image")
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)

# in_ds = gdal.Open(r"H:\xzr\duxiang\压缩文件\201912_GF2.img")
# in_band1 = in_ds.GetRasterBand(1)
#
# dst_ds = gdal.GetDriverByName('GTiff').Create("hello1.tif", 147474, 78234, 1, in_band1.DataType)
# dst_ds.SetGeoTransform(in_ds.GetGeoTransform())
# dst_ds.SetProjection(in_ds.GetProjection())
