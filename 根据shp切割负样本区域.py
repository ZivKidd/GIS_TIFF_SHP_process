import os

import cv2
import gdal
import numpy as np
import shapefile
import tqdm
# import arcpy
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


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

    # 负样本shp有多个区域，遍历shp文件


if __name__ == "__main__":
    sf = shapefile.Reader(r"D:\Desktop\负样本区域范围shp\负样本区域范围.shp")  # 读取未识别目标物的shp文件
    dataset = gdal.Open(r"J:\山西整幅\影像\山西全省20200508.img")
    # dataset_mask = gdal.Open(r"J:\shanxibianpo\bianpo.tif")
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    # im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    # im_proj = dataset.GetProjection()  # 地图投影信息

    in_band1 = dataset.GetRasterBand(1)
    in_band2 = dataset.GetRasterBand(2)
    in_band3 = dataset.GetRasterBand(3)

    output_folder = r"J:\shanxibianpo\FYB_img"
    shapes = sf.shapes()
    for i in tqdm.tqdm(range(len(shapes))):
        # if(i>5):
        #     break

        shp = shapes[i]  # 获取shp文件中的每一个形状

        point = shp.points  # 获取shp的每一个点
        x_list = np.asarray([ii[0] for ii in point])
        y_list = np.asarray([ii[1] for ii in point])

        coords = geo2imagexy(dataset, x_list, y_list)
        x_max = np.floor(max(coords[0, :]))
        y_max = np.floor(max(coords[1, :]))
        x_min = np.floor(min(coords[0, :]))
        y_min = np.floor(min(coords[1, :]))
        offset_x = int(np.floor(x_max - x_min))
        offset_y = int(np.floor(y_max - y_min))
        out_band1 = in_band1.ReadAsArray(x_min, y_min, offset_x, offset_y)
        out_band2 = in_band2.ReadAsArray(x_min, y_min, offset_x, offset_y)
        out_band3 = in_band3.ReadAsArray(x_min, y_min, offset_x, offset_y)
        # out_bandmask = mask_band.ReadAsArray(offset_x, offset_y, cell, cell)

        # out_bandmask[np.where(out_bandmask > 0)] = 255

        out_band1 = np.reshape(out_band1, [out_band1.shape[0], out_band1.shape[1], 1])
        out_band2 = np.reshape(out_band2, [out_band2.shape[0], out_band2.shape[1], 1])
        out_band3 = np.reshape(out_band3, [out_band3.shape[0], out_band3.shape[1], 1])
        image = np.concatenate([out_band3, out_band2, out_band1], axis=2)
        cv2.imwrite(os.path.join(output_folder, 'FYB_{0}_{1}.png'.format(offset_x, offset_y)), image)
