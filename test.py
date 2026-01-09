# %%
import kagglehub
import shutil
import os

# 1. Download to the default cache (returns the local path)
cache_path = kagglehub.dataset_download("mohankrishnathalla/diabetes-health-indicators-dataset")

# 2. Define your destination and move the files
dest_path = "data/raw"
os.makedirs(dest_path, exist_ok=True)

# Move all files from cache to your local folder
for file in os.listdir(cache_path):
    shutil.move(os.path.join(cache_path, file), os.path.join(dest_path, file))

print(f"Dataset moved to: {dest_path}")