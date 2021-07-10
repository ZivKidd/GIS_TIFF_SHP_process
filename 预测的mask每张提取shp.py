# -*- coding: utf-8 -*-
import glob
import json
import os

import cv2
import gdal
import numpy as np
import tqdm


# 每张mask影像中提取边界，转换到大影像坐标系再转换到经纬度

def imagexy2geo(dataset, row, col):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
    trans = dataset.GetGeoTransform()
    px = trans[0] + row * trans[1] + col * trans[2]
    py = trans[3] + row * trans[4] + col * trans[5]
    return px, py


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


if __name__ == '__main__':
    original_tifs = [r"J:\山西整幅\影像\山西全省20200508.img"]
    pred_jsons = [r"J:\shanxibianpo\bianpo_predv2.geojson"]
    mask_pngs = [r"J:\shanxibianpo\Val\shanxibianpo1024__*_*_yanshou.png"]

    for j in range(len(original_tifs)):

        result = []
        num = 0
        in_ds = gdal.Open(original_tifs[j])
        im_width = in_ds.RasterXSize  # 栅格矩阵的列数
        im_height = in_ds.RasterYSize  # 栅格矩阵的行数

        # dst_ds = gdal.GetDriverByName('GTiff').Create(pred_tifs[j], im_width, im_height, 1, gdal.GDT_Byte)
        # dst_ds.SetGeoTransform(in_ds.GetGeoTransform())
        # dst_ds.SetProjection(in_ds.GetProjection())

        masks = glob.glob(mask_pngs[j])
        # masks = []

        for i, m in enumerate(tqdm.tqdm(masks)):

            name = os.path.split(m)[1][:-4]
            name = name.split('_')
            x0 = int(name[-3])
            y0 = int(name[-2])

            img = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
            if (np.sum(img) < 1):
                continue
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            # cv2.imshow('ori',img)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            img = cv2.erode(img, kernel)  # 腐蚀图像
            img = cv2.erode(img, kernel)  # 腐蚀图像
            # cv2.imshow('erode',img)

            # img = cv2.dilate(img, kernel)  # 膨胀图像
            img = cv2.dilate(img, kernel)  # 膨胀图像
            if (np.sum(img) < 1):
                continue

            # print m
            contours, hierarch = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for c in contours:
                rect = {}

                rect['type'] = "Feature"

                points = c.reshape([-1, 2]).astype(np.float)

                points[:, 0] += y0
                points[:, 1] += x0

                points = np.asarray(imagexy2geo(in_ds, points[:, 0], points[:, 1]))
                # points[:, 0] = points[:, 0] * x_delta + x0
                # points[:, 1] = points[:, 1] * y_delta + y0
                points = points.T
                points = points.tolist()
                points = [points]

                rect["geometry"] = {}
                rect["geometry"]["type"] = 'Polygon'
                rect["geometry"]["coordinates"] = points

                rect["properties"] = {}
                rect['properties']['index'] = num
                rect["properties"]["disaster_type"] = 'huapo'

                num += 1
                result.append(rect)

        result1 = {}
        result1['type'] = 'FeatureCollection'
        result1['name'] = 'duxiang_polygonV1'
        result1["crs"] = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
        result1['features'] = result

        with open(pred_jsons[j], 'w') as f:
            json.dump(result1, f)
        # print 1
