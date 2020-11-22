import glob
import shutil
import os
files=glob.glob(r"H:\xzr\process\result_good\nishiliu\*\mask\*.png")
for f in files:
    name=os.path.split(f)[1]
    shutil.copyfile(f,r"H:\xzr\process\result_good\nishiliu\all\mask\\"+name)
print()