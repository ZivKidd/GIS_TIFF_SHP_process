# -*- coding: utf-8 -*-
import gdal
import osgeo
import os
import shutil
import shapefile
from osgeo import osr
import numpy as np
import arcpy
from PIL import Image, ImageDraw

arcpy.env.workspace = r"D:\xzr\process\data.gbd"  # arcgis地理数据库目录


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


def getFileName(file_dir):
    file_path_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if (file[-3:] == "tif"):
                file_path_list.append(unicode(root + '\\' + file,'gbk'))
    return file_path_list


tif_folders = [r"D:\deepleearning\sampleshp\tif_all"]
tif_files = []
for t in tif_folders:
    tif_files.extend(getFileName(t))

shp_path = r"D:\deepleearning\sampleshp\titian\titian.shp"  # shp文件的路径， shapefile不支持中文路径
out_dir = r"D:\deepleearning\sampleshp\titian\tif_clip"  # 裁剪后图像保存路径
sf = shapefile.Reader(shp_path)  # 读取shp文件
shapes = sf.shapes()

for i in range(len(shapes)):
    print str(i) + '/' + str(len(shapes))
    shp = shapes[i]  # 获取shp文件中的每一个形状

    point = shp.points  # 获取每一个最小外接矩形的四个点
    x_list = [ii[0] for ii in point]
    y_list = [ii[1] for ii in point]

    x_min = min(x_list)
    y_min = min(y_list)
    x_max = max(x_list)
    y_max = max(y_list)

    x_cen = (x_min + x_max) / 2
    y_cen = (y_max + y_min) / 2

    x_min1 = x_min - (x_max - x_min) / 2
    x_max1 = x_max + (x_max - x_min) / 2
    y_min1 = y_min - (y_max - y_min) / 2
    y_max1 = y_max + (y_max - y_min) / 2
    #
    # x_min1 = x_min - (x_max - x_min) * 2
    # x_max1 = x_max + (x_max - x_min) * 2
    # y_min1 = y_min - (y_max - y_min) * 2
    # y_max1 = y_max + (y_max - y_min) * 2

    # x_min1 = x_min - (x_max - x_min)
    # x_max1 = x_max + (x_max - x_min)
    # y_min1 = y_min - (y_max - y_min)
    # y_max1 = y_max + (y_max - y_min)

    count = -1
    for t in tif_files:
        dataset = gdal.Open(t)
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数

        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_proj = dataset.GetProjection()  # 地图投影信息

        im_x_min = im_geotrans[0]
        im_y_max = im_geotrans[3]
        im_x_max = im_x_min + im_width * im_geotrans[1]
        im_y_min = im_y_max + im_height * im_geotrans[5]

        if (y_cen < im_y_max and y_cen > im_y_min and x_cen < im_x_max and x_cen > im_x_min):

            coords = lonlat2imagexy(dataset, x_cen, y_cen)
            coords1 = lonlat2geo(dataset, x_max1, y_max1)
            coords2 = lonlat2geo(dataset, x_min1, y_min1)
            x_min2 = coords2[0]
            y_min2 = coords2[1]
            x_max2 = coords1[0]
            y_max2 = coords1[1]

            if (coords[0] < im_width and coords[1] < im_height):
                print t
                count += 1

                out_path = os.path.join(out_dir, str(i) + '-' + str(count) + '.tif')

                cor = str(x_min2) + ' ' + str(y_min2) + ' ' + str(x_max2) + ' ' + str(y_max2)
                # cor = str(x_min1) + ' ' + str(y_min1) + ' ' + str(x_max1) + ' ' + str(y_max1)
                print cor
                arcpy.Clip_management(t, cor, out_path, "#", "#", None)  # 调用工具箱函数

                try:
                    img = Image.open(out_path)
                    print 'good'
                except:
                    os.remove(out_path)
                    os.remove(out_path + '.ovr')
                    print 'deleted'
