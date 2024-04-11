import os, datetime
import sys
from pathlib import Path
import shutil
from distutils.dir_util import copy_tree

# import datasets
os.system("gcloud storage cp gs://uw-nn-storage/DATASET_T2.csv /home/deamoon_uw_nn/bucket_source")
os.system("gcloud storage cp gs://uw-nn-storage/DATASET_V2.csv /home/deamoon_uw_nn/bucket_source")

# remove & pull to update main
dirpath = Path('/home/deamoon_uw_nn') / 'uw-nn-adam'
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
        
os.system("git clone https://github.com/Deam0on/uw-nn-adam.git")

# run main
os.system("python3 /home/deamoon_uw_nn/uw-nn-adam/main.py")

# update bucket - results to resuls (new) and archive all
os.system("gcloud storage cp /home/deamoon_uw_nn/bucket_source gs://uw-nn-storage/Results --recursive")

#create archive
mydir = Path('/home/deamoon_uw_nn') / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs(mydir)

#
copy_tree("/home/deamoon_uw_nn/bucket_source", "mydir")
os.system("gcloud storage cp mydir gs://uw-nn-storage/Archive --recursive")
shutil.rmtree(mydir)
