# -*- coding: utf-8 -*-
import os
import shutil
import arcpy
import shapefile

arcpy.env.workspace = r"D:\xzr\landslide\process\data.gbd"  # arcgis地理数据库目录
shp_path = r"H:\xzr\process\dataxiluodu\Export_Output.shp"  # shp文件的路径， shapefile不支持中文路径
raster = r"H:\xzr\溪洛渡水电站沿线滑坡\truecolor\GF2_PMS1_E103.1_N27.4_20170109_L1A0002107346-PAN1_ortho_fuse_color.tif"  # 图像路径
out_dir = r"H:\xzr\process\dataxiluodu\tif"  # 裁剪后图像保存路径
# if os.path.exists(out_dir):
#     shutil.rmtree(out_dir)
# os.mkdir(out_dir)

sf = shapefile.Reader(shp_path)  # 读取shp文件
shapes = sf.shapes()

for i in range(len(shapes)):

    shp = shapes[i]  # 获取shp文件中的每一个形状

    out_path = os.path.join(out_dir,str(i) + '-1.tif')

    point = shp.points  # 获取每一个最小外接矩形的四个点
    x_list = [ii[0] for ii in point]
    y_list = [ii[1] for ii in point]

    x_min = min(x_list)
    y_min = min(y_list)
    x_max = max(x_list)
    y_max = max(y_list)

    x_min1 = x_min - (x_max - x_min) * 2
    x_max1 = x_max + (x_max - x_min) * 2
    y_min1 = y_min - (y_max - y_min) * 2
    y_max1 = y_max + (y_max - y_min) * 2

    cor = str(x_min1) + ' ' + str(y_min1) + ' ' + str(x_max1) + ' ' + str(y_max1)
    arcpy.Clip_management(raster, cor, out_path, "#", "#", None)  # 调用工具箱函数

    print i

