from PIL import Image, ImageDraw
import os
try:
    img = Image.open(r"H:\xzr\process\data2014\huapo\tif\0-0.tif")
except:
    os.remove(r"H:\xzr\process\data2014\huapo\tif\0-0.tif")
print 0