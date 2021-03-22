import glob
import os

import cv2
import numpy as np
import tqdm

files = glob.glob(r"I:\20141215dagangshan\result\all\mask\*.png")
folder = r"I:\20141215dagangshan\result\all\mask_white"
for f in tqdm.tqdm(files):
    img = cv2.imread(f)
    img = np.sum(img, axis=-1)
    img = np.reshape(img, [img.shape[0], img.shape[1], 1])
    img = np.tile(img, [1, 1, 3]).astype(np.int)
    name = os.path.split(f)[1]
    new_path = os.path.join(folder, name)
    cv2.imwrite(new_path, img)
    # print()
