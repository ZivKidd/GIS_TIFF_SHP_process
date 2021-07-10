# -*- coding: utf-8 -*-
import os

import cv2
import gdal
import numpy as np
import shapefile
import tqdm
# import arcpy
from PIL import Image
from osgeo import osr

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


sf = shapefile.Reader(r"J:\shanxibianpo\undect_bianpo.shp")  # 读取未识别目标物的shp文件
dataset = gdal.Open(r"J:\山西整幅\影像\山西全省20200508.img")
# dataset_mask = gdal.Open(r"J:\shanxibianpo\bianpo.tif")
im_width = dataset.RasterXSize  # 栅格矩阵的列数
im_height = dataset.RasterYSize  # 栅格矩阵的行数
# im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
# im_proj = dataset.GetProjection()  # 地图投影信息

in_band1 = dataset.GetRasterBand(1)
in_band2 = dataset.GetRasterBand(2)
in_band3 = dataset.GetRasterBand(3)
# mask_band = dataset_mask.GetRasterBand(1)

output_folder = r"J:\shanxibianpo\undect_img"
# output_folder_mask = r"J:\shanxibianpo\mask"
cell = 1024

shapes = sf.shapes()
for i in tqdm.tqdm(range(len(shapes))):
    # if(i>5):
    #     break

    shp = shapes[i]  # 获取shp文件中的每一个形状

    point = shp.points  # 获取每一个最小外接矩形的四个点
    x_list = [ii[0] for ii in point]
    y_list = [ii[1] for ii in point]

    # if (isOutOfRaster(x_list, y_list, raster_x_min, raster_x_max, raster_y_min, raster_y_max)):
    #     continue

    x_min = min(x_list)
    y_min = min(y_list)
    x_max = max(x_list)
    y_max = max(y_list)

    x_cen = (x_min + x_max) / 2
    y_cen = (y_max + y_min) / 2

    # if (not use_proj_coord):
    #     coords = lonlat2imagexy(dataset, x_cen, y_cen)
    # else:
    coords = geo2imagexy(dataset, x_cen, y_cen)
    coords = (int(round(abs(coords[0]))), int(round(abs(coords[1]))))

    offset_x = coords[0] - cell / 2
    offset_y = coords[1] - cell / 2

    # i=0
    # while(i+cell<im_height):
    #     print datetime.datetime.now()
    #     print i
    #     print im_height
    #     j = 0
    #     while(j+cell<im_width):
    # print j
    # print im_width
    out_band1 = in_band1.ReadAsArray(offset_x, offset_y, cell, cell)
    out_band2 = in_band2.ReadAsArray(offset_x, offset_y, cell, cell)
    out_band3 = in_band3.ReadAsArray(offset_x, offset_y, cell, cell)
    # out_bandmask = mask_band.ReadAsArray(offset_x, offset_y, cell, cell)

    # out_bandmask[np.where(out_bandmask > 0)] = 255

    out_band1 = np.reshape(out_band1, [out_band1.shape[0], out_band1.shape[1], 1])
    out_band2 = np.reshape(out_band2, [out_band2.shape[0], out_band2.shape[1], 1])
    out_band3 = np.reshape(out_band3, [out_band3.shape[0], out_band3.shape[1], 1])
    image = np.concatenate([out_band3, out_band2, out_band1], axis=2)

    # if(np.where(out_band1==0)[0].shape[0]+np.where(out_band1==255)[0].shape[0]!=cell**2):
    cv2.imwrite(os.path.join(output_folder, 'shanxi_{0}_{1}.png'.format(offset_x, offset_y)), image)
    # cv2.imwrite(os.path.join(output_folder_mask, 'shanxi_{0}_{1}.png'.format(offset_x, offset_y)), out_bandmask)
    #
    #     j+=cell
    # i+=cell
