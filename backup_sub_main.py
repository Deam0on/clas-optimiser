import os
from datetime import datetime
import sys
from pathlib import Path
import shutil
from distutils.dir_util import copy_tree

# import datasets
os.system("gcloud storage cp gs://uw-nn-storage_v2/Dataset/DATASET_T2.csv /home/deamoon_uw_nn/bucket_source")
os.system("gcloud storage cp gs://uw-nn-storage_v2/Dataset/DATASET_V2.csv /home/deamoon_uw_nn/bucket_source")

# remove & pull to update main
dirpath = Path('/home/deamoon_uw_nn') / 'uw-nn-adam'
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
        
os.system("git clone https://github.com/Deam0on/uw-nn-adam.git")

# run main
os.system("python3 /home/deamoon_uw_nn/uw-nn-adam/main.py")

# update bucket - results to resuls (new) and archive all
os.system("gsutil cp -r /home/deamoon_uw_nn/bucket_source gs://uw-nn-storage_v2/Results")

#create archive
today = datetime.now()
new_run = r"/home/deamoon_uw_nn/{}" .format(today.strftime('%Y-%m-%d_%H-%M-%S'))
os.mkdir(new_run)

#copy and remove
copy_tree("/home/deamoon_uw_nn/bucket_source", new_run)
os.system("gsutil cp -r "+ new_run +" gs://uw-nn-storage_v2/Archive")
shutil.rmtree(new_run)
