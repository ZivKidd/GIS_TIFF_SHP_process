# -*- coding: utf-8 -*-
import gdal
import osgeo
import os
import shutil
import shapefile
from osgeo import osr
import numpy as np
# import arcpy
from PIL import Image, ImageDraw
import sys
Image.MAX_IMAGE_PIXELS = None

# arcpy.env.workspace = r"D:\xzr\process\data.gbd"  # arcgis地理数据库目录


def lonlat2geo(dataset, lon, lat):
    '''
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]


def getSRSPair(dataset):
    '''
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs


def geo2imagexy(dataset, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解


def lonlat2imagexy(dataset, x, y):
    '''
    影像行列转经纬度：
    ：通过经纬度转平面坐标
    ：平面坐标转影像行列
    '''
    coords = lonlat2geo(dataset, x, y)
    coords2 = geo2imagexy(dataset, coords[0], coords[1])
    return (int(round(abs(coords2[0]))), int(round(abs(coords2[1]))))


shp_path =r"H:\xzr\duxiang_buffer\newest-huapo\newest.shp" # # shp文件的路径， shapefile不支持中文路径
# out_dir = r"H:\xzr\duxiang_buffer\real-huapo\tif\\"  # 裁剪后图像保存路径

dataset = gdal.Open(r"H:\xzr\duxiang_buffer\2kmclip.tif")
im_width = dataset.RasterXSize  # 栅格矩阵的列数
im_height = dataset.RasterYSize  # 栅格矩阵的行数
im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
im_proj = dataset.GetProjection()  # 地图投影信息

in_band1 = dataset.GetRasterBand(1)
in_band2 = dataset.GetRasterBand(2)
in_band3 = dataset.GetRasterBand(3)


sf = shapefile.Reader(shp_path)  # 读取shp文件
shapes = sf.shapes()


# raster = np.zeros((im_height,im_width), dtype=np.int)
mask_all=Image.new('RGB',(im_width,im_height))
color=(255, 255, 255)

for i in range(len(shapes)):
    print (str(i) + '/' + str(len(shapes)))
    shp = shapes[i]  # 获取shp文件中的每一个形状
    point = shp.points  # 获取每一个最小外接矩形的四个点
    x_list = [ii[0] for ii in point]
    y_list = [ii[1] for ii in point]

    vertice = []

    for j in range(len(x_list)-1):
        coords = lonlat2imagexy(dataset, x_list[j], y_list[j])
        vertice.append(coords)

    draw = ImageDraw.Draw(mask_all)
    # 滑坡
    draw.polygon(vertice, fill=color)

cell=1024
w=int(im_width/(cell/2))-1
h=int(im_height/(cell/2))-1
for i in range(w):
    for j in range(h):
        # 定义切图的起始点坐标(相比原点的横坐标和纵坐标偏移量)
        offset_x =int(i*cell/2)
        offset_y =int(j*cell/2)

        mask = mask_all.crop((offset_x, offset_y, offset_x + cell, offset_y + cell))
        mask.save(r"H:\xzr\duxiang_buffer\real-huapo\clip_1024\\" +str(offset_x)+'-'+str(offset_y)+ '.png')

        # img = np.array(mask)

        # if(np.where(img==0)[0].shape[0]>1024**2/2):
        #     continue

        print(offset_x,offset_y)
