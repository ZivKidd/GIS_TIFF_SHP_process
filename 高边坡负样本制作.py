import datetime
import os

import cv2
import gdal
import numpy as np
import shapefile
import tqdm
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


# 读取文件夹中负样本区域图像文件然后切割
def cut_FYBArea_FromImg(img_folder, save_folder):
    fyb_dir = sorted([f for f in os.listdir(img_folder) if f.endswith(".png")])
    print(fyb_dir)
    num_fybs = len(fyb_dir)

    fyb_savefolder = os.path.join(save_folder, 'fyb')
    label_savefolder = os.path.join(save_folder, 'label')
    if not os.path.isdir(fyb_savefolder):
        os.makedirs(fyb_savefolder)
    if not os.path.isdir(label_savefolder):
        os.makedirs(label_savefolder)

    num = 1  # 负样本的数量
    for k in range(num_fybs):
        imgPath = fyb_dir[k]
        # 读取图片
        fybimg = cv2.imread(os.path.join(img_folder, imgPath))
        im_height, im_width, Channel = fybimg.shape
        fyblabel = np.zeros([im_height, im_width, 1])  # 负样本的标签
        cell = 1024
        i = 0
        while (i + cell < im_height):
            print(datetime.datetime.now())
            print(i)
            print(im_height)
            j = 0
            while (j + cell < im_width):
                fyb = fybimg[i:i + cell, j:j + cell, :]
                label = fyblabel[i:i + cell, j:j + cell, :]
                cv2.imwrite(os.path.join(fyb_savefolder, 'fyb{0}.png'.format(num)), fyb)
                cv2.imwrite(os.path.join(label_savefolder, 'fyb{0}.png'.format(num)), label)
                num += 1
                j += cell
            i += cell


# 有一些容易被错分的shp单独剪切下来
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


def get_FYBImage_FromShp(shp_path, save_folder):
    sf = shapefile.Reader(shp_path)  # 读取未识别目标物的shp文件
    dataset = gdal.Open(r"J:\山西整幅\影像\山西全省20200508.img")
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    in_band1 = dataset.GetRasterBand(1)
    in_band2 = dataset.GetRasterBand(2)
    in_band3 = dataset.GetRasterBand(3)

    fyb_savefolder = os.path.join(save_folder, 'fyb')
    label_savefolder = os.path.join(save_folder, 'label')
    num = len(os.listdir(fyb_savefolder))  # 统计原有负样本的数量
    num += 1
    cell = 1024

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

        x_cen = (x_min + x_max) / 2
        y_cen = (y_max + y_min) / 2

        # if (not use_proj_coord):
        #     coords = lonlat2imagexy(dataset, x_cen, y_cen)
        # else:
        coords = geo2imagexy(dataset, x_cen, y_cen)
        coords = (int(round(abs(coords[0]))), int(round(abs(coords[1]))))

        offset_x = coords[0] - cell / 2
        offset_y = coords[1] - cell / 2

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
        cv2.imwrite(os.path.join(fyb_savefolder, 'fyb{0}.png'.format(num)), image)
        label = np.zeros([cell, cell, 1])
        cv2.imwrite(os.path.join(label_savefolder, 'fyb{0}.png'.format(num)), label)
        num += 1


def get_RoadBufferFYB_FromShp(shp_path, save_folder):
    sf = shapefile.Reader(shp_path)  # 读取未识别目标物的shp文件
    dataset = gdal.Open(r"J:\山西整幅\影像\山西全省20200508.img")
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    in_band1 = dataset.GetRasterBand(1)
    in_band2 = dataset.GetRasterBand(2)
    in_band3 = dataset.GetRasterBand(3)

    fyb_savefolder = os.path.join(save_folder, 'fyb')
    label_savefolder = os.path.join(save_folder, 'label')
    if not os.path.isdir(fyb_savefolder):
        os.makedirs(fyb_savefolder)
    if not os.path.isdir(label_savefolder):
        os.makedirs(label_savefolder)
    num = 882  # 统计原有负样本的数量
    num += 1
    cell = 1024

    shapes = sf.shapes()
    for i in tqdm.tqdm(range(len(shapes))):
        shp = shapes[i]  # 获取shp文件中的每一个形状

        point = shp.points  # 获取每一个最小外接矩形的四个点
        # x_list = np.asarray([ii[0] for ii in point])
        # y_list = np.asarray([ii[1] for ii in point])
        #
        # coords = geo2imagexy(dataset, x_list, y_list)
        # x_max = np.floor(max(coords[0, :]))
        # y_max = np.floor(max(coords[1, :]))
        # x_min = np.floor(min(coords[0, :]))
        # y_min = np.floor(min(coords[1, :]))
        # offset_x = int(np.floor(x_max - x_min))
        # offset_y = int(np.floor(y_max - y_min))
        # out_band1 = in_band1.ReadAsArray(x_min, y_min, offset_x, offset_y)
        # out_band2 = in_band2.ReadAsArray(x_min, y_min, offset_x, offset_y)
        # out_band3 = in_band3.ReadAsArray(x_min, y_min, offset_x, offset_y)
        #
        # out_band1 = np.reshape(out_band1, [out_band1.shape[0], out_band1.shape[1], 1])
        # out_band2 = np.reshape(out_band2, [out_band2.shape[0], out_band2.shape[1], 1])
        # out_band3 = np.reshape(out_band3, [out_band3.shape[0], out_band3.shape[1], 1])
        # image = np.concatenate([out_band3, out_band2, out_band1], axis=2)
        #
        # new_image = np.zeros([cell, cell, 3])
        # cen = cell//2
        # new_image[int(cen-offset_y/2):int(cen+offset_y/2),int(cen-offset_x/2):int(cen+offset_x/2),:] = image

        # if(np.where(out_band1==0)[0].shape[0]+np.where(out_band1==255)[0].shape[0]!=cell**2):

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

        out_band1 = in_band1.ReadAsArray(offset_x, offset_y, cell, cell)
        out_band2 = in_band2.ReadAsArray(offset_x, offset_y, cell, cell)
        out_band3 = in_band3.ReadAsArray(offset_x, offset_y, cell, cell)
        # out_bandmask = mask_band.ReadAsArray(offset_x, offset_y, cell, cell)

        # out_bandmask[np.where(out_bandmask > 0)] = 255

        out_band1 = np.reshape(out_band1, [out_band1.shape[0], out_band1.shape[1], 1])
        out_band2 = np.reshape(out_band2, [out_band2.shape[0], out_band2.shape[1], 1])
        out_band3 = np.reshape(out_band3, [out_band3.shape[0], out_band3.shape[1], 1])
        image = np.concatenate([out_band3, out_band2, out_band1], axis=2)
        # new_image = np.zeros([2*cell, 2*cell, 3])
        # cen = cell*2//2
        # new_image[int(cen-cell/2):int(cen+cell/2),int(cen-cell/2):int(cen+cell/2),:] = image

        cv2.imwrite(os.path.join(fyb_savefolder, 'fyb{0}.png'.format(num)), image)
        label = np.zeros([cell, cell, 1])
        cv2.imwrite(os.path.join(label_savefolder, 'fyb{0}.png'.format(num)), label)
        num += 1


if __name__ == "__main__":
    fyb_folder = r"J:\shanxibianpo\FYB_img"  # 负样本区域影像文件夹
    save_folder = r"D:\semantic-segmentation-for-Geographical-survey\Slope3\FYB"  # 负样本保存文件夹
    cut_FYBArea_FromImg(fyb_folder, save_folder)

    fyb_ShpfilePath = r"D:\Desktop\负样本shp\负样本.shp"
    get_FYBImage_FromShp(fyb_ShpfilePath, save_folder)
    # fyb_roadbufferShpPath = r"D:\Desktop\500m缓冲区负样本\cuowu\FYB_buffer_predict_bianpo_predv6.shp"
    # get_RoadBufferFYB_FromShp(fyb_roadbufferShpPath, save_folder)
