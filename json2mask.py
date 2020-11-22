import glob
import json
import os
json_files=glob.glob(r'H:\无人机\loudi\娄底json\*.json')
for j in json_files:
    with open(j) as f:
        js=json.load(f)

        print()