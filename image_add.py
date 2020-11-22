import gdal
import numpy as np
import os
from PIL import Image, ImageDraw
import datetime
import cv2
import json

def getAllFileName(folder_path):
    file_list = []
    folder_list = []
    for file_name in os.listdir(folder_path):
        if (os.path.isfile(os.path.join(folder_path, file_name))):
            file_list.append(os.path.join(folder_path, file_name))
        elif (os.path.isdir(os.path.join(folder_path, file_name))):
            folder_list.append(os.path.join(folder_path, file_name))
    file_list.sort()
    return file_list, folder_list

tif_folder=r"H:\xzr\nas_uav\Giulia\DOM-UAV\UAV DOM\clip_2048\\"
file_list, _ = getAllFileName(r"D:\semantic-segmentation-for-Geographical-survey\Val1")
result_folder=r"H:\xzr\nas_uav\Giulia\DOM-UAV\UAV DOM\add_2048\\"
for i,file in enumerate(file_list):
    print(i,len(file_list))

    file_name = os.path.split(file)[1][:-4]
    tif_name=tif_folder+file_name+'.tif'
    tif = cv2.imread(tif_name, cv2.IMREAD_COLOR)


    img = cv2.imread(file, cv2.IMREAD_COLOR)
    img=cv2.resize(img,(2048,2048))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    image, contours, hierarch = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if (len(contours) == 0):
        continue
    list1 = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        # if (area < 4000):
        #     continue
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(tif,(int(x),int(y)),(int(x+w),int(y+h)),(0,0,255),5)



    # result2 = cv2.add(img, tif)
    # cv2.imshow(file_name, result2)
    # cv2.waitKey()
    # print()

    cv2.imwrite(result_folder+file_name+'.png',tif)