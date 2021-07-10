import json

import numpy as np
import shapefile
import tqdm
from shapely.geometry import Polygon

# 中咨标注样本
gt_shp_path = r"J:\shanxibianpo\all.shp"
# 预测样本
predict_shp_path = r"J:\shanxibianpo\bianpo_predv2.shp"
# 未识别
pred_jsons = [r"J:\shanxibianpo\unbianpo_predv3.geojson"]

sf = shapefile.Reader(predict_shp_path)  # 读取预测的shp文件
shapes = sf.shapes()

gt_sf = shapefile.Reader(gt_shp_path)  # 读取中资公司标注shp文件
gt_shapes = gt_sf.shapes()
all_num = len(gt_shapes)

num = 0
result = []
k = 0

for i, gt_shp in enumerate(tqdm.tqdm(gt_shapes)):
    gt_points = np.asarray(gt_shp.points)
    # 找出真实的样本多边形的X坐标最小值和最大值
    # lsP = gt_points
    # gt_polygon_x_max, gt_polygon_x_min, gt_y_temp = max(lsP)[0], min(lsP)[0], [i[1] for i in lsP]
    gt_polygon_x_max = np.max(gt_points[:, 0])
    gt_polygon_y_max = np.max(gt_points[:, 1])
    gt_polygon_x_min = np.min(gt_points[:, 0])
    gt_polygon_y_min = np.min(gt_points[:, 1])

    gt_plg = Polygon(gt_points)
    tag = False  # 判断是否检测出来

    for j, shp in enumerate(shapes):
        points = np.asarray(shp.points)

        # lsP = points
        polygon_x_max = np.max(points[:, 0])
        polygon_x_min = np.min(points[:, 0])
        polygon_y_max = np.max(points[:, 1])
        polygon_y_min = np.min(points[:, 1])
        # polygon_y_max, polygon_y_min, y_temp = max(lsP)[0], min(lsP)[0], [i[1] for i in lsP]

        if ((polygon_x_max < gt_polygon_x_min) or (polygon_x_min > gt_polygon_x_max) or (
                polygon_y_min > gt_polygon_y_max) or (polygon_y_max < gt_polygon_y_min)):
            continue
        plg = Polygon(points)

        if (gt_plg.intersects(plg)):
            num += 1
            tag = True
            break

    if (tag == False):  # 如果检测不出来则执行
        rect = {}

        rect['type'] = "Feature"

        points = gt_points.tolist()
        points = [points]
        rect["geometry"] = {}
        rect["geometry"]["type"] = 'Polygon'
        rect["geometry"]["coordinates"] = points

        rect["properties"] = {}
        rect['properties']['index'] = num
        rect["properties"]["disaster_type"] = 'huapo'
        result.append(rect)

result1 = {}
result1['type'] = 'FeatureCollection'
result1['name'] = 'duxiang_polygonV1'
result1["crs"] = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
result1['features'] = result

with open(pred_jsons[k], 'w') as f:
    json.dump(result1, f)

print(num / all_num)
