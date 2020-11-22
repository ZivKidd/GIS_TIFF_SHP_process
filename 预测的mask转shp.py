import gdal
import numpy as np
import os
import datetime
import cv2
import json
import glob
import tqdm

# 先把所有预测的图片合成到一个大影像
# 再在里面找边界

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

if __name__=='__main__':

    in_ds = gdal.Open(r"H:\xzr\duxiang_buffer\2kmclip.tif")
    in_band1 = in_ds.GetRasterBand(1)

    gt = in_ds.GetGeoTransform()
    x0 = gt[0]
    x_delta = gt[1]
    y0 = gt[3]
    y_delta = gt[-1]

    im_width = in_ds.RasterXSize  # 栅格矩阵的列数
    im_height = in_ds.RasterYSize  # 栅格矩阵的行数
    im_geotrans = in_ds.GetGeoTransform()  # 仿射矩阵
    im_proj = in_ds.GetProjection()  # 地图投影信息

    raster_x_min, raster_x_max, raster_y_min, raster_y_max = raster_boarder(im_geotrans, im_width, im_height)
    print(raster_x_min, raster_x_max, raster_y_min, raster_y_max)

    # in_band1 = dataset.GetRasterBand(1)
    # in_band2 = dataset.GetRasterBand(2)
    # in_band3 = dataset.GetRasterBand(3)


    # raster = np.zeros((im_height,im_width), dtype=np.int)

    masks=glob.glob(r"D:\semantic-segmentation-for-Geographical-survey\Val1\*.png")

    mask_all = np.zeros([im_height,im_width],dtype=np.uint8)

    for i,m in enumerate(tqdm.tqdm(masks)):
        # print(i)
        img = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
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

        # cv2.imshow('dilate',img)

        # cv2.waitKey(0)

        # image, contours, hierarch = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if (np.sum(img) < 1):
            continue
        name = os.path.split(m)[1][:-4]
        name = name.split('-')
        x1 = int(name[0])
        y1 = int(name[1])
        valid_xy=np.where(img>0)
        valid_x=valid_xy[0]+y1
        valid_y=valid_xy[1]+x1

        mask_all[valid_x,valid_y]=255

        # print(np.sum(img),np.sum(mask_all))

        # print()

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, kernel)
    # mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, kernel)
    # print(datetime.datetime.now())
    image, contours, hierarch = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(datetime.datetime.now())
    # print(contours)
    num=1

    result=[]
    for c in contours:
        rect={}

        rect['type']= "Feature"


        points=c.reshape([-1, 2]).astype(np.float)
        points[:,0]=points[:,0]*x_delta+x0
        points[:,1]=points[:,1]*y_delta+y0

        points=points.tolist()
        points=[points]

        rect["geometry"]={}
        rect["geometry"]["type"]='Polygon'
        rect["geometry"]["coordinates"]=points

        rect["properties"]={}
        rect['properties']['index']=num
        rect["properties"]["disaster_type"] = 'huapo'

        num+=1
        result.append(rect)

    result1={}
    result1['type']='FeatureCollection'
    result1['name']='duxiang_polygonV1'
    result1["crs"]= { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } }
    result1['features']=result

    with open('duxiang_polygonV1.geojson', 'w') as f:
        json.dump(result1, f)


    # gtif_driver = gdal.GetDriverByName("GTiff")
    # # 创建切出来的要存的文件（3代表3个不都按，最后一个参数为数据类型，跟原文件一致）
    # out_ds = gtif_driver.Create('mask_all.tif', im_width,im_height,  1,
    #                             in_band1.DataType)
    # # print("create new tif file succeed")
    #
    # # 获取原图的原点坐标信息
    # ori_transform = in_ds.GetGeoTransform()
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
    # # top_left_x = top_left_x + offset_x * w_e_pixel_resolution
    # # top_left_y = top_left_y + offset_y * n_s_pixel_resolution
    #
    # # 将计算后的值组装为一个元组，以方便设置
    # dst_transform = (top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])
    #
    # # 设置裁剪出来图的原点坐标
    # out_ds.SetGeoTransform(dst_transform)
    #
    # # 设置SRS属性（投影信息）
    # out_ds.SetProjection(in_ds.GetProjection())
    #
    # # 写入目标文件
    # out_ds.GetRasterBand(1).WriteArray(mask_all)
    # # out_ds.GetRasterBand(2).WriteArray(out_band2)
    # # out_ds.GetRasterBand(3).WriteArray(out_band3)
    #
    # # 将缓存写入磁盘
    # out_ds.FlushCache()
    # print("FlushCache succeed")


    # result=[]
    # result0=[]
    # num=1
    # file_list, _ = getAllFileName(r"D:\semantic-segmentation-for-Geographical-survey\Val1")
    # for file in file_list:
    #
    #     img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    #     if (np.sum(img) < 1):
    #         continue
    #
    #     name = os.path.split(file)[1][:-4]
    #     name = name.split('-')
    #     x1 = int(name[0])
    #     y1 = int(name[1])
    #
    #     # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    #     img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #     img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #     img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    #
    #     image, contours, hierarch = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     if (len(contours) == 0):
    #         continue
    #
    #     list1 = []
    #     for i in range(len(contours)):
    #         area = cv2.contourArea(contours[i])
    #         if (area < 1200):
    #             continue
    #         x, y, w, h = cv2.boundingRect(contours[i])
    #         print(x, y, w, h)
    #         print(area)
    #
    #         xl = x1 + x-2
    #         yt = y1 + y+2
    #         xr = xl + w
    #         yd = yt + h
    #
    #         xlc = xl * x_delta + x0
    #         xrc = xr * x_delta + x0
    #         ytc = yt * y_delta + y0
    #         ydc = yd * y_delta + y0
    #
    #         result0.append([xlc,xrc,ytc,ydc])
    #
    # for _ in range(5):
    #     i=0
    #     while i<len(result0):
    #         j=i+1
    #         while j < len(result0):
    #             if(isIntersection(result0[i][0],result0[i][1],result0[i][3],result0[i][2],result0[j][0],result0[j][1],result0[j][3],result0[j][2])):
    #                 xl_new=min(result0[i][0],result0[j][0])
    #                 xr_new=max(result0[i][1],result0[j][1])
    #                 yd_new=min(result0[i][3],result0[j][3])
    #                 yt_new=max(result0[i][2],result0[j][2])
    #                 result0.append([xl_new, xr_new, yt_new, yd_new])
    #                 result0.remove(result0[i])
    #                 result0.remove(result0[j-1])
    #                 j-=1
    #             j+=1
    #         i+=1
    #     print(len(result0))
    #
    #
    # for xlc,xrc,ytc,ydc in result0:
    #     rect={}
    #
    #     rect['type']= "Feature"
    #     rect['properties']={'index':num}
    #
    #     points=[[[xlc,ytc],[xlc,ydc],[xrc,ydc],[xrc,ytc],[xlc,ytc]]]
    #
    #     rect["geometry"]={}
    #     rect["geometry"]["type"]='Polygon'
    #     rect["geometry"]["coordinates"]=points
    #
    #
    #
    #     num+=1
    #     result.append(rect)
    #
    # result1={}
    # result1['type']='FeatureCollection'
    # result1['name']='duxiangV19'
    # result1["crs"]= { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } }
    # result1['features']=result
    #
    # with open('duxiangV19.geojson', 'w') as f:
    #     json.dump(result1, f)
    # print(0)

            # cv2.namedWindow("Image")
            # cv2.imshow("Image", img)
            # cv2.waitKey(0)

    # in_ds = gdal.Open(r"H:\xzr\duxiang\压缩文件\201912_GF2.img")
    # in_band1 = in_ds.GetRasterBand(1)
    #
    # dst_ds = gdal.GetDriverByName('GTiff').Create("hello1.tif", 147474, 78234, 1, in_band1.DataType)
    # dst_ds.SetGeoTransform(in_ds.GetGeoTransform())
    # dst_ds.SetProjection(in_ds.GetProjection())
