# -*- coding: utf-8 -*-
# encoding: utf-8

# 输入：所有的tif路径和shp文件的路径
# 输出：png格式的image和mask


import sys

reload(sys)

sys.setdefaultencoding('utf8')
import gdal
import os
import shapefile
from osgeo import osr
import numpy as np
# import arcpy
from PIL import Image
import tqdm
import cv2

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


# 获取影像边界
def raster_boarder(im_geotrans, im_width, im_height):
    raster_x_min = im_geotrans[0]
    raster_x_max = raster_x_min + im_width * im_geotrans[1]
    raster_y_max = im_geotrans[3]
    raster_y_min = raster_y_max + im_height * im_geotrans[5]

    # raster_x_min+=1024*im_geotrans[1]
    # raster_x_max-=1024*im_geotrans[1]
    # raster_y_max+=1024*im_geotrans[5]
    # raster_y_min-=1024*im_geotrans[5]

    return raster_x_min, raster_x_max, raster_y_min, raster_y_max


# 迭代获取路径下所有tif影像
def getFileName(file_dir):
    file_path_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if (file[-3:] == "tif"):
                file_path_list.append((root + '\\' + file))
    return file_path_list


# 看一个polygon的xy是否与影像完全没有交集
def isOutOfRaster(x_list, y_list, raster_x_min, raster_x_max, raster_y_min, raster_y_max):
    x_max = max(x_list)
    y_max = max(y_list)
    x_min = min(x_list)
    y_min = min(y_list)
    if (x_max < raster_x_min):
        return True
    if (y_max < raster_y_min):
        return True
    if (y_min > raster_y_max):
        return True
    if (x_min > raster_x_max):
        return True
    return False


# 需要改的参数
#########################################################################################################
for date_string in ['huapo', 'nishiliu', 'bengta']:
    # date_string = 'huapo'  # 文件的独特命名
    num = 0
    shp_path = r"I:\20141215dagangshan\shapefiles\\" + date_string + ".shp"  # shp文件的路径， shapefile不支持中文路径
    tif_file = r"I:\20141215dagangshan\all\3_dsm_ortho\2_mosaic\all_transparent_mosaic_group1.tif"
    tif_file_mask = r"I:\20141215dagangshan\result\all.tif"
    all_dir = r"I:\20141215dagangshan\result\all_bb\image\\"
    all_dir_mask = r"I:\20141215dagangshan\result\all_bb\mask\\"
    use_proj_coord = True
    print_shp = False
    out_dir = all_dir + "_tif"  # 裁剪后图像保存路径
    dataset = gdal.Open(tif_file)
    dataset_mask = gdal.Open(tif_file_mask)
    # im_width = dataset.RasterXSize  # 栅格矩阵的列数
    # im_height = dataset.RasterYSize  # 栅格矩阵的行数
    # im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    # im_proj = dataset.GetProjection()  # 地图投影信息
    #
    # raster_x_min, raster_x_max, raster_y_min, raster_y_max = raster_boarder(im_geotrans, im_width, im_height)
    # print(raster_x_min, raster_x_max, raster_y_min, raster_y_max)
    #
    in_band1 = dataset.GetRasterBand(1)
    in_band2 = dataset.GetRasterBand(2)
    in_band3 = dataset.GetRasterBand(3)
    in_band_mask = dataset_mask.GetRasterBand(1)

    sf = shapefile.Reader(shp_path)  # 读取shp文件
    shapes = sf.shapes()
    for i in tqdm.tqdm(range(len(shapes))):
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

        # x_cen = (x_min + x_max) / 2
        # y_cen = (y_max + y_min) / 2

        if (not use_proj_coord):
            (x_min, y_min) = lonlat2imagexy(dataset, x_min, y_min)
            (x_max, y_max) = lonlat2imagexy(dataset, x_max, y_max)
        else:
            # coords = geo2imagexy(dataset, x_cen, y_cen)
            (x_min, y_min) = geo2imagexy(dataset, x_min, y_min)
            (x_max, y_max) = geo2imagexy(dataset, x_max, y_max)
            # (x_min,y_min) = lonlat2imagexy(dataset, x_min,y_min)
            # (x_max, y_max) = lonlat2imagexy(dataset, x_max, y_max)
            x_min = int(round(abs(x_min)))
            y_min = int(round(abs(y_min)))
            x_max = int(round(abs(x_max)))
            y_max = int(round(abs(y_max)))

        # offset_x = coords[0] - cell / 2
        # offset_y = coords[1] - cell / 2
        #
        # if (offset_x < 0 or offset_y < 0 or offset_x + cell > im_width or offset_y + cell > im_height):
        #     continue
        result_x_min = min(x_min, x_max)
        result_x_max = max(x_min, x_max)
        result_y_min = min(y_min, y_max)
        result_y_max = max(y_min, y_max)
        out_band1 = in_band1.ReadAsArray(result_x_min, result_y_min, result_x_max - result_x_min,
                                         result_y_max - result_y_min)
        out_band2 = in_band2.ReadAsArray(result_x_min, result_y_min, result_x_max - result_x_min,
                                         result_y_max - result_y_min)
        out_band3 = in_band3.ReadAsArray(result_x_min, result_y_min, result_x_max - result_x_min,
                                         result_y_max - result_y_min)
        out_band_mask = in_band_mask.ReadAsArray(result_x_min, result_y_min, result_x_max - result_x_min,
                                                 result_y_max - result_y_min)

        out_band1 = np.reshape(out_band1, [out_band1.shape[0], out_band1.shape[1], 1])
        out_band2 = np.reshape(out_band2, [out_band2.shape[0], out_band2.shape[1], 1])
        out_band3 = np.reshape(out_band3, [out_band3.shape[0], out_band3.shape[1], 1])
        image = np.concatenate([out_band1, out_band2, out_band3], axis=2)

        mask = np.zeros(image.shape)

        mask[np.where(out_band_mask == 1)] = np.array([255, 0, 0])
        mask[np.where(out_band_mask == 2)] = np.array([0, 255, 0])
        mask[np.where(out_band_mask == 3)] = np.array([0, 0, 255])

        num += 1
        cv2.imwrite(all_dir + '\\' + date_string + '-' + str(num) + '.png', image)
        cv2.imwrite(all_dir_mask + '\\' + date_string + '-' + str(num) + '.png', mask)

        # print
        # num += 1
        # # 获取Tif的驱动，为创建切出来的图文件做准备
        # gtif_driver = gdal.GetDriverByName("GTiff")
        #
        # # 创建切出来的要存的文件（3代表3个不都按，最后一个参数为数据类型，跟原文件一致）
        # out_ds = gtif_driver.Create(out_dir + '\\' + date_string + '-' + str(num) + '.tif', cell, cell, 3,
        #                             in_band1.DataType)
        # # print("create new tif file succeed")
        #
        # # 获取原图的原点坐标信息
        # ori_transform = dataset.GetGeoTransform()
        # # if ori_transform:
        # #     print (ori_transform)
        # #     print("Origin = ({}, {})".format(ori_transform[0], ori_transform[3]))
        # #     print("Pixel Size = ({}, {})".format(ori_transform[1], ori_transform[5]))
        #
        # # 读取原图仿射变换参数值
        # top_left_x = ori_transform[0]  # 左上角x坐标
        # w_e_pixel_resolution = ori_transform[1]  # 东西方向像素分辨率
        # top_left_y = ori_transform[3]  # 左上角y坐标
        # n_s_pixel_resolution = ori_transform[5]  # 南北方向像素分辨率
        #
        # # 根据反射变换参数计算新图的原点坐标
        # top_left_x = top_left_x + offset_x * w_e_pixel_resolution
        # top_left_y = top_left_y + offset_y * n_s_pixel_resolution
        #
        # # 将计算后的值组装为一个元组，以方便设置
        # dst_transform = (top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])
        #
        # # 设置裁剪出来图的原点坐标
        # out_ds.SetGeoTransform(dst_transform)
        #
        # # 设置SRS属性（投影信息）
        # out_ds.SetProjection(dataset.GetProjection())
        #
        # # 写入目标文件
        # out_ds.GetRasterBand(1).WriteArray(out_band1)
        # out_ds.GetRasterBand(2).WriteArray(out_band2)
        # out_ds.GetRasterBand(3).WriteArray(out_band3)
        #
        # # 将缓存写入磁盘
        # out_ds.FlushCache()
        #
        # img = Image.open(out_dir + '\\' + date_string + '-' + str(num) + '.tif')
        # img.save(all_dir + '\\' + date_string + '-' + str(num) + '.png')
