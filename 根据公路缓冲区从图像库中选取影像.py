import glob
import os
import shutil

import gdal
import numpy as np
import shapefile
import tqdm
from PIL import Image
from osgeo import osr
from shapely.geometry import Point
from shapely.geometry import Polygon

Image.MAX_IMAGE_PIXELS = None


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


if __name__ == '__main__':
    original_tifs = [r"J:\山西整幅\影像\山西全省20200508.img"]  # 读取全省影像
    buffer_shp_path = r"J:\shanxibianpo\shanxi_Road500mBuffer_test.shp"  # 缓冲区的shp文件
    imgOverlap_pngs = [r"J:\shanxibianpo\image1024overlap\shanxibianpo1024overlap__*_*.png"]
    savefolder = r"J:\shanxibianpo\image1024buffer_new"
    sf = shapefile.Reader(buffer_shp_path)  # 读取缓冲区的shp文件
    bf_shp = sf.shapes()[0]  # 缓冲区只有一个

    cell = 1024

    for j in range(len(original_tifs)):
        in_ds = gdal.Open(original_tifs[j])
        im_width = in_ds.RasterXSize  # 栅格矩阵的列数
        im_height = in_ds.RasterYSize  # 栅格矩阵的行数
        imgdata = glob.glob(imgOverlap_pngs[j])

        bf_points = np.asarray(bf_shp.points)

        bf_polygon_x_max = np.max(bf_points[:, 0])  # 经度
        bf_polygon_y_max = np.max(bf_points[:, 1])  # 纬度
        bf_polygon_x_min = np.min(bf_points[:, 0])
        bf_polygon_y_min = np.min(bf_points[:, 1])

        bf_plg = Polygon(bf_points)

        for i, m in enumerate(tqdm.tqdm(imgdata)):
            name = os.path.split(m)[1][:-4]
            name = name.split('_')
            # 左上角顶点的坐标
            x0 = int(name[2])
            y0 = int(name[3])
            # 右下角顶点的坐标
            x1 = x0 + cell
            y1 = y0 + cell
            # 左下角坐标
            x2 = x0 + cell
            y2 = y0
            # 右上角坐标
            x3 = x0
            y3 = y0 + cell
            # 生成多边形为顺时针方向
            points = np.array([[y0, x0], [y3, x3], [y1, x1], [y2, x2]])

            points = np.asarray(imagexy2geo(in_ds, points[:, 0], points[:, 1]))
            points = points.T  # 转置
            polygon_x_max = np.max(points[:, 0])
            polygon_x_min = np.min(points[:, 0])
            polygon_y_max = np.max(points[:, 1])
            polygon_y_min = np.min(points[:, 1])

            if ((polygon_x_max < bf_polygon_x_min) or (polygon_x_min > bf_polygon_x_max) or (
                    polygon_y_min > bf_polygon_y_max) or (polygon_y_max < bf_polygon_y_min)):
                continue

            zs_point = Point(points[0])
            yx_point = Point(points[1])
            zx_point = Point(points[2])
            ys_point = Point(points[3])
            # m_plg = Polygon(points) #通过四个点构造多边形
            # if(bf_plg.intersects(m_plg)):
            #     shutil.move(m, savefolder)
            # 判断点是否在多边形内部或者边上
            if (zs_point.within(bf_plg) or zs_point.within(bf_plg.boundary) or yx_point.within(
                    bf_plg) or yx_point.within(bf_plg.boundary)
                    or zx_point.within(bf_plg) or zx_point.within(bf_plg.boundary) or ys_point.within(
                        bf_plg) or ys_point.within(bf_plg.boundary)):
                shutil.move(m, savefolder)
