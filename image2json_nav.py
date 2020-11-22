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

def isIntersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    intersect_flag = True

    minx = max(xmin_a , xmin_b)
    miny = max(ymin_a , ymin_b)

    maxx = min(xmax_a , xmax_b)
    maxy = min(ymax_a , ymax_b)
    if minx > maxx or miny > maxy:
        intersect_flag = False
    return intersect_flag


in_ds = gdal.Open(r"H:\xzr\nas_uav\Giulia\DOM-UAV\UAV DOM\DOM_2014.tif")

gt = in_ds.GetGeoTransform()
x0 = gt[0]
x_delta = gt[1]
y0 = gt[3]
y_delta = gt[-1]

result=[]
result0=[]
num=1
file_list, _ = getAllFileName(r"D:\semantic-segmentation-for-Geographical-survey\Val1")
for file in file_list:

    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if (np.sum(img) < 0):
        continue

    name = os.path.split(file)[1][:-4]
    name = name.split('-')
    x1 = int(name[0])
    y1 = int(name[1])

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_NEAREST)

    image, contours, hierarch = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if (len(contours) == 0):
        continue

    list1 = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if (area < 4000):
            continue
        x, y, w, h = cv2.boundingRect(contours[i])
        print(x, y, w, h)
        print(area)

        xl = x1 + x-2
        yt = y1 + y+2
        xr = xl + w
        yd = yt + h

        xlc = xl * x_delta + x0
        xrc = xr * x_delta + x0
        ytc = yt * y_delta + y0
        ydc = yd * y_delta + y0

        result0.append([xlc,xrc,ytc,ydc])

for _ in range(5):
    i=0
    while i<len(result0):
        j=i+1
        while j < len(result0):
            if(isIntersection(result0[i][0],result0[i][1],result0[i][3],result0[i][2],result0[j][0],result0[j][1],result0[j][3],result0[j][2])):
                xl_new=min(result0[i][0],result0[j][0])
                xr_new=max(result0[i][1],result0[j][1])
                yd_new=min(result0[i][3],result0[j][3])
                yt_new=max(result0[i][2],result0[j][2])
                result0.append([xl_new, xr_new, yt_new, yd_new])
                result0.remove(result0[i])
                result0.remove(result0[j-1])
                j-=1
            j+=1
        i+=1
    print(len(result0))


for xlc,xrc,ytc,ydc in result0:
    rect={}

    rect['type']= "Feature"
    rect['properties']={'index':num}

    points=[[[xlc,ytc],[xlc,ydc],[xrc,ydc],[xrc,ytc],[xlc,ytc]]]

    rect["geometry"]={}
    rect["geometry"]["type"]='Polygon'
    rect["geometry"]["coordinates"]=points



    num+=1
    result.append(rect)

result1={}
result1['type']='FeatureCollection'
result1['name']='navV1'

result1['features']=result

with open('navV1.geojson', 'w') as f:
    json.dump(result1, f)
print(0)

        # cv2.namedWindow("Image")
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)

# in_ds = gdal.Open(r"H:\xzr\duxiang\压缩文件\201912_GF2.img")
# in_band1 = in_ds.GetRasterBand(1)
#
# dst_ds = gdal.GetDriverByName('GTiff').Create("hello1.tif", 147474, 78234, 1, in_band1.DataType)
# dst_ds.SetGeoTransform(in_ds.GetGeoTransform())
# dst_ds.SetProjection(in_ds.GetProjection())
