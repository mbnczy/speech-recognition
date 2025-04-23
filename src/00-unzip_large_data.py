import os
import zipfile
import argparse
import logging
import time
from tqdm import tqdm
from pathlib import Path
import shutil


def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.{decimal_places}f} {unit}"
        size /= 1024.0

def folder_size_and_file_count(folder_path):
    total_size = 0
    file_count = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
            file_count += 1
    return total_size, file_count
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_folder', type=str, default=os.environ.get("ZIP_FOLDER", "../large_data"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    zip_files = [filename for filename in os.listdir(args.zip_folder) if filename.endswith('.zip')]

    logging.info(f"Extracting {len(zip_files)} zip files.")

    for zip_file in tqdm(zip_files, desc="Unzipping"):
        start_time = time.time()
        target_dir = os.path.join(args.zip_folder, os.path.splitext(zip_file)[0])
        os.makedirs(target_dir, exist_ok=True)

        #unzip
        with zipfile.ZipFile(os.path.join(args.zip_folder, zip_file), 'r') as zip_ref:
            file_list = zip_ref.namelist()
            for file in tqdm(file_list, desc=f"Extracting {zip_file}", leave=False):
                zip_ref.extract(member=file, path=target_dir)

        duration = time.time() - start_time
        size, count = folder_size_and_file_count(target_dir)

        logging.info(f"Extracted '{zip_file}' to '{target_dir}' in {duration:.2f}s | "
                     f"Size: {human_readable_size(size)} | Files: {count}")