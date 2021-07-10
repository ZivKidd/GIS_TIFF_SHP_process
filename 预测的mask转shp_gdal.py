# -*- coding: utf-8 -*-
import glob
import os

import cv2
import gdal
import numpy as np
import ogr
import tqdm


# 先把所有预测的图片合成到一个大影像
# 再在里面找边界

# 获取影像边界
def raster_boarder(im_geotrans, im_width, im_height):
    raster_x_min = im_geotrans[0]
    raster_x_max = raster_x_min + im_width * im_geotrans[1]
    raster_y_max = im_geotrans[3]
    raster_y_min = raster_y_max + im_height * im_geotrans[5]

    return raster_x_min, raster_x_max, raster_y_min, raster_y_max


if __name__ == '__main__':
    original_tifs = [r"J:\shanxibianpo\road_buffer.tif"]
    pred_tifs = [r"J:\shanxibianpo\buffer_predict_bianpo_predv_newV2.tif"]
    pred_shps = [r"J:\shanxibianpo\buffer_predict_bianpo_predv_newV2.shp"]
    mask_pngs = [r"J:\shanxibianpo\imagebuffer_Val\shanxibianpo1024buffer__*_*.png"]
    cell = 1024
    for j in range(len(original_tifs)):

        in_ds = gdal.Open(original_tifs[j])

        im_width = in_ds.RasterXSize  # 栅格矩阵的列数
        im_height = in_ds.RasterYSize  # 栅格矩阵的行数

        dst_ds = gdal.GetDriverByName('GTiff').Create(pred_tifs[j], im_width, im_height, 1, gdal.GDT_Byte)
        dst_ds.SetGeoTransform(in_ds.GetGeoTransform())
        dst_ds.SetProjection(in_ds.GetProjection())

        masks = glob.glob(mask_pngs[j])
        # masks=[]
        # for i,m in enumerate(tqdm.tqdm(masks1)):
        #     # try:
        #     name = os.path.split(m)[1][:-4]
        #     name = name.split('_')
        #     x1 = int(name[-3])
        #     y1 = int(name[-2])
        #     if(x1>500000 or x1<300000):
        #         continue
        #     masks.append(m)

        for i, m in enumerate(tqdm.tqdm(masks)):
            # try:
            name = os.path.split(m)[1][:-4]
            name = name.split('_')
            x1 = int(name[2])
            y1 = int(name[3])
            # if(y1>500000 or y1<300000):
            #     continue
            img = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
            if (np.sum(img) < 1):
                continue
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            img = cv2.erode(img, kernel)  # 腐蚀图像
            img = cv2.erode(img, kernel)  # 腐蚀图像
            # cv2.imshow('erode',img)

            # img = cv2.dilate(img, kernel)  # 膨胀图像
            img = cv2.dilate(img, kernel)  # 膨胀图像

            if (np.sum(img) < 1):
                continue

            # 由于部分有重叠的,一张图上有，另一张图没有，所以
            old_img = dst_ds.GetRasterBand(1).ReadAsArray(y1, x1, cell, cell)
            img = old_img + img
            img[np.where(img > 1)] = 255

            dst_ds.GetRasterBand(1).WriteArray(img, y1, x1)
            # except:
            #     continue
        dst_ds.FlushCache()

        sourceRaster = gdal.Open(pred_tifs[j])
        outShapefile = pred_shps[j]
        # outShapefileGood = r"I:\weiyigaosu\all\yuce1python_good.shp"

        band = sourceRaster.GetRasterBand(1)
        driver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(outShapefile):
            driver.DeleteDataSource(outShapefile)
        outDatasource = driver.CreateDataSource(outShapefile)
        outLayer = outDatasource.CreateLayer("polygonized", srs=None)
        dst_fieldname = 'DN'
        fd = ogr.FieldDefn('DN', ogr.OFTInteger)
        outLayer.CreateField(fd)
        # 添加面积字段 用于将面积小于100的删掉
        new_field = ogr.FieldDefn("Area", ogr.OFTReal)
        new_field.SetWidth(32)
        new_field.SetPrecision(16)  # 设置面积精度,小数点后16位
        outLayer.CreateField(new_field)

        gdal.Polygonize(band, None, outLayer, 0, [], callback=None)

        sourceRaster = None

        # inds = ogr.Open(outShapefile)
        # inlayer = inds.GetLayer()
        feature = outLayer.GetNextFeature()
        while feature:
            cover = feature.GetField('DN')
            # 计算面积
            geom = feature.GetGeometryRef()
            area = geom.GetArea()  # 面积的单位是度，转为平方米为单位
            m_area = (area / (0.0089 ** 2)) * 1e+6  # 单位由十进制度转为米
            if (cover == 0 or m_area <= 130):
                id = feature.GetFID()
                outLayer.DeleteFeature(id)
            feature.Destroy()
            feature = outLayer.GetNextFeature()
        outDatasource.Destroy()

# def area(shpPath):
#     '''计算面积'''
#     driver = ogr.GetDriverByName("ESRI Shapefile")
#     dataSource = driver.Open(shpPath, 1)
#     layer = dataSource.GetLayer()
#     new_field = ogr.FieldDefn("Area", ogr.OFTReal)
#     new_field.SetWidth(32)
#     new_field.SetPrecision(16)  # 设置面积精度,小数点后16位
#     layer.CreateField(new_field)
#     for feature in layer:
#         geom = feature.GetGeometryRef()
#         area = geom.GetArea()  # 计算面积
#         # m_area = (area/(0.0089**2))*1e+6  # 单位由十进制度转为米
#         # print(m_area)
#         feature.SetField("Area", area)  # 将面积添加到属性表中
#         layer.SetFeature(feature)
#     dataSource = None
