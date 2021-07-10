# -*- coding: utf-8 -*-
import os

import gdal
import ogr


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


if __name__ == '__main__':
    sourceRaster = gdal.Open(r"I:\weiyigaosu\all\yuce4.tif")
    outShapefile = r"I:\weiyigaosu\all\yuce4.shp"
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
    gdal.Polygonize(band, None, outLayer, 0, [], callback=None)

    sourceRaster = None

    # inds = ogr.Open(outShapefile)
    # inlayer = inds.GetLayer()
    feature = outLayer.GetNextFeature()
    while feature:
        cover = feature.GetField('DN')
        if cover == 0:
            id = feature.GetFID()
            outLayer.DeleteFeature(id)
        feature.Destroy()
        feature = outLayer.GetNextFeature()
    outDatasource.Destroy()

    # sf = shapefile.Reader(outShapefile)  # 读取shp文件
    # shapeRec = sf.shapeRecords()
    #
    #
    # w = shapefile.Writer(outShapefileGood, shapeType=sf.shapeType)
    # w.field('FIRST_FLD', 'C', '40')
    #
    #
    # for i in range(len(shapeRec)):
    #     shp = shapeRec[i].shape
    #     rec=shapeRec[i].record[0]
    #     if(rec==255):
    #         w.poly([shp.points])
    #         w.record('FIRST_FLD', str(i))
    # w.close()
