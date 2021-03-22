# -*- coding: utf-8 -*-
import os

import cv2
import gdal
import numpy as np
import shapefile
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


shp_paths = [r"I:\dem_fenlei\rs_wanli_huapo_bengta\bengta.shp",
             r"I:\dem_fenlei\rs_wanli_huapo_bengta\huapo_good.shp",
             r"I:\dem_fenlei\rs_ludian_huapo_bengta\bengta_valid.shp",
             r"I:\dem_fenlei\rs_ludian_huapo_bengta\huapo_valid.shp",
             r"I:\dem_fenlei\rs_uav_dagangshan_huapo_bengta_nishiliu\bengta4326.shp",
             r"I:\dem_fenlei\rs_uav_dagangshan_huapo_bengta_nishiliu\huapo4326.shp"]
tif_paths = []
tif_paths.append(
    [r"I:\dem_fenlei\rs_wanli_huapo_bengta\dem_bengta.tif", r"I:\dem_fenlei\rs_wanli_huapo_bengta\slope_bengta.tif"])
tif_paths.append(
    [r"I:\dem_fenlei\rs_wanli_huapo_bengta\dem_huapo.tif", r"I:\dem_fenlei\rs_wanli_huapo_bengta\slope_huapo.tif"])
tif_paths.append(
    [r"I:\dem_fenlei\rs_ludian_huapo_bengta\dem_bengta.tif", r"I:\dem_fenlei\rs_ludian_huapo_bengta\slope_bengta.tif"])
tif_paths.append(
    [r"I:\dem_fenlei\rs_ludian_huapo_bengta\dem_huapo.tif", r"I:\dem_fenlei\rs_ludian_huapo_bengta\slope_huapo.tif"])
tif_paths.append([r"I:\dem_fenlei\rs_uav_dagangshan_huapo_bengta_nishiliu\dem_bengta.tif",
                  r"I:\dem_fenlei\rs_uav_dagangshan_huapo_bengta_nishiliu\slope_bengta.tif"])
tif_paths.append([r"I:\dem_fenlei\rs_uav_dagangshan_huapo_bengta_nishiliu\dem_huapo.tif",
                  r"I:\dem_fenlei\rs_uav_dagangshan_huapo_bengta_nishiliu\slope_huapo.tif"])

for i in range(len(shp_paths)):
    print i
    shp_path = shp_paths[i]
    dem_path = tif_paths[i][0]
    slope_path = tif_paths[i][1]
    print shp_path
    print dem_path
    print slope_path
    txt_path = shp_path[:-4] + '.csv'

    dataset_dem = gdal.Open(dem_path)
    in_band_dem = dataset_dem.GetRasterBand(1)
    pixel_size = dataset_dem.GetGeoTransform()[1] * 10000
    dataset_slope = gdal.Open(slope_path)
    in_band_slope = dataset_slope.GetRasterBand(1)

    sf = shapefile.Reader(shp_path)  # 读取shp文件
    shapeRec = sf.shapeRecords()

    with open(txt_path, 'w') as f:
        name = '周长,面积,最小外接矩形长，最小外接矩形宽，最大高程，最小高程，高差，高度平均值，高度中位数，高度标准差，最大坡度，最小坡度，坡度差，坡度平均值，坡度中位数，坡度标准差，坡度10分位内平均，坡度20分位内平均，坡度30分位内平均，坡度40分位内平均，坡度50分位内平均，坡度60分位内平均，坡度70分位内平均，坡度80分位内平均，坡度90分位内平均，坡度100分位内平均'
        name = name.replace('，', ',')
        f.write(name + '\n')
        for i in range(len(shapeRec)):
            shp = shapeRec[i].shape  # 获取shp文件中的每一个形状

            point = shp.points  # 获取每一个最小外接矩形的四个点
            x_list = [ii[0] for ii in point]
            y_list = [ii[1] for ii in point]

            x_min = min(x_list)
            y_min = min(y_list)
            x_max = max(x_list)
            y_max = max(y_list)

            coords_min = lonlat2imagexy(dataset_dem, x_min, y_min)
            coords_max = lonlat2imagexy(dataset_dem, x_max, y_max)

            all_cluster = 0

            out_band_dem = dataset_dem.ReadAsArray(coords_min[0], coords_max[1], int(coords_max[0] - coords_min[0]),
                                                   int(coords_min[1] - coords_max[1]))
            out_band_slope = dataset_slope.ReadAsArray(coords_min[0], coords_max[1], int(coords_max[0] - coords_min[0]),
                                                       int(coords_min[1] - coords_max[1]))
            if (out_band_dem is None):
                continue
            if (np.where(out_band_dem > 0)[0].size == 0):
                continue
            out_band_binary = np.zeros(out_band_slope.shape, dtype=np.uint8)
            out_band_binary[np.where(out_band_dem > 0)] = 255

            # 周长
            contours = cv2.findContours(out_band_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 得到轮廓信息
            cnt = contours[0][0]  # 取第一条轮廓
            perimeter = cv2.arcLength(cnt, True) * pixel_size
            # 面积
            area = cv2.contourArea(cnt) * pixel_size * pixel_size
            # 最小外接矩形长
            bbox = cv2.minAreaRect(cnt)
            bbox_length = np.max(bbox[1]) * pixel_size
            # 最小外接矩形宽
            bbox_width = np.min(bbox[1]) * pixel_size
            # 最大高程
            dem_valid = out_band_dem[np.where(out_band_dem > 0)]
            elevation_max = np.max(dem_valid)
            # 最小高程
            elevation_min = np.min(dem_valid)
            # 高差
            elevation_diff = elevation_max - elevation_min
            # 高度平均值
            elevation_mean = np.mean(dem_valid)
            # 高度中位数
            elevation_median = np.median(dem_valid)
            # 高度标准差
            elevation_std = np.std(dem_valid)
            # 最大坡度
            slope_valid = out_band_slope[np.where(out_band_dem > 0)]
            slope_max = np.max(slope_valid)
            # 最小坡度
            slope_min = np.min(slope_valid)
            # 坡度差
            slope_diff = slope_max - slope_min
            # 坡度平均值
            slope_mean = np.mean(slope_valid)
            # 坡度中位数
            slope_median = np.median(slope_valid)
            # 坡度标准差
            slope_std = np.std(slope_valid)
            # 坡度10分位数
            values = np.sort(slope_valid)
            values_0_10 = np.mean(values[int(values.shape[0] * 0):int(values.shape[0] * 0.1)])
            values_10_20 = np.mean(values[int(values.shape[0] * 0.1):int(values.shape[0] * 0.2)])
            values_20_30 = np.mean(values[int(values.shape[0] * 0.2):int(values.shape[0] * 0.3)])
            values_30_40 = np.mean(values[int(values.shape[0] * 0.3):int(values.shape[0] * 0.4)])
            values_40_50 = np.mean(values[int(values.shape[0] * 0.4):int(values.shape[0] * 0.5)])
            values_50_60 = np.mean(values[int(values.shape[0] * 0.5):int(values.shape[0] * 0.6)])
            values_60_70 = np.mean(values[int(values.shape[0] * 0.6):int(values.shape[0] * 0.7)])
            values_70_80 = np.mean(values[int(values.shape[0] * 0.7):int(values.shape[0] * 0.8)])
            values_80_90 = np.mean(values[int(values.shape[0] * 0.8):int(values.shape[0] * 0.9)])
            values_90_100 = np.mean(values[int(values.shape[0] * 0.9):int(values.shape[0] * 1)])

            # name = '周长,面积,最小外接矩形长，最小外接矩形宽，最大高程，最小高程，高差，高度平均值，高度中位数，' \
            #        '高度标准差，最大坡度，最小坡度，坡度差，坡度平均值，坡度中位数，坡度标准差，坡度10分位内平均，' \
            #        '坡度20分位内平均，坡度30分位内平均，坡度40分位内平均，坡度50分位内平均，坡度60分位内平均，' \
            #        '坡度70分位内平均，坡度80分位内平均，坡度90分位内平均，坡度100分位内平均'

            f.write(str(perimeter))
            f.write(',')
            f.write(str(area))
            f.write(',')
            f.write(str(bbox_length))
            f.write(',')
            f.write(str(bbox_width))
            f.write(',')
            f.write(str(elevation_max))
            f.write(',')
            f.write(str(elevation_min))
            f.write(',')
            f.write(str(elevation_diff))
            f.write(',')
            f.write(str(elevation_mean))
            f.write(',')
            f.write(str(elevation_median))
            f.write(',')
            f.write(str(elevation_std))
            f.write(',')
            f.write(str(slope_max))
            f.write(',')
            f.write(str(slope_min))
            f.write(',')
            f.write(str(slope_diff))
            f.write(',')
            f.write(str(slope_mean))
            f.write(',')
            f.write(str(slope_median))
            f.write(',')
            f.write(str(slope_std))
            f.write(',')
            f.write(str(values_0_10))
            f.write(',')
            f.write(str(values_10_20))
            f.write(',')
            f.write(str(values_20_30))
            f.write(',')
            f.write(str(values_30_40))
            f.write(',')
            f.write(str(values_40_50))
            f.write(',')
            f.write(str(values_50_60))
            f.write(',')
            f.write(str(values_60_70))
            f.write(',')
            f.write(str(values_70_80))
            f.write(',')
            f.write(str(values_80_90))
            f.write(',')
            f.write(str(values_90_100))
            f.write('\n')
        # print
