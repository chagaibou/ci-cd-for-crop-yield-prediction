import shutil
from pathlib import Path

DATASET_TYPES = ["test", "train"]
TARGET_COLUMN = "crop_yield"
RAW_DATASET = "dataset/crop_yield_data.csv"



def delete_and_recreate_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    finally:
        Path(path).mkdir(parents=True, exist_ok=True)
