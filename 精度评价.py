import shapefile

def isOverlap(point,gt_point):
    # 多边形坐标值
    lsP = point
    polygon_x_max, polygon_x_min, y_temp = max(lsP)[0], min(lsP)[0], [i[1] for i in lsP]
    polygon_y_max, polygon_y_min = max(y_temp), min(y_temp)
    # # 矩形坐标值
    # rectangle_x_max, rectangle_x_min = 1117, 785
    # rectangle_y_max, rectangle_y_min = 716, 366
    # rectangle_points = [[rectangle_x_min, rectangle_y_min], [rectangle_x_max, rectangle_y_min],
    #                     [rectangle_x_min, rectangle_y_max], [rectangle_x_max, rectangle_y_max]]

    rectangle_points=gt_point
    rectangle_x_max, rectangle_x_min, y_temp = max(rectangle_points)[0], min(rectangle_points)[0], [i[1] for i in rectangle_points]
    rectangle_y_max, rectangle_y_min = max(y_temp), min(y_temp)

    oddNodes = False
    # 判断矩形与多边形控制矩形是否相交
    # for point in rectangle_points:
    if (polygon_x_min < rectangle_x_max < polygon_x_max or polygon_x_min < rectangle_x_min < polygon_x_max) and (
            polygon_y_min < rectangle_y_min < polygon_y_max or polygon_y_min < rectangle_y_max < polygon_y_max):
        # 判断矩形顶点与多边形边的关系
        for point in rectangle_points:
            try:
                x, y = point[0], point[1]
                j = len(lsP) - 1
                for i in range(len(lsP)):
                    if lsP[i][1] < y < lsP[j][1] or lsP[j][1] < y < lsP[i][1] and x >= lsP[i][0] or x >= lsP[j][0]:
                        if (lsP[i][0] + (y - lsP[i][1]) / (lsP[j][1] - lsP[i][1]) * (lsP[j][0] - lsP[i][0])) < x:
                            oddNodes = True
                    j = i
            except:
                continue
    return oddNodes


shp_path=r"D:\xzr\shp-process\duxiangV17-2.shp"
# shp_path=r"D:\xzr\shp-process\duxiangV15-9.shp"
gt_shp_path=r"H:\xzr\duxiang_buffer\newest-huapo\newest.shp"

sf = shapefile.Reader(shp_path)  # 读取shp文件
shapes = sf.shapes()

gt_sf=shapefile.Reader(gt_shp_path)
gt_shapes=gt_sf.shapes()

for i,shp in enumerate(shapes):
    # print (str(i) + '/' + str(len(shapes)))
    # shp = shapes[i]  # 获取shp文件中的每一个形状
    point = shp.points  # 获取每一个最小外接矩形的四个点
    # # point_np=
    # x_list = [ii[0] for ii in point]
    # y_list = [ii[1] for ii in point]

    for j,gt_shp in enumerate(gt_shapes):
        gt_point=gt_shp.points

        if(isOverlap(point,gt_point)):
            print(i,j)
        # gt_x_list=[ii[0] for ii in gt_point]
        # gt_y_list=[ii[1] for ii in gt_point]
        #
        # print()