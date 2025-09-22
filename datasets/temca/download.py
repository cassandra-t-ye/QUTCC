import os
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# SETTINGS
URL = "https://temca2data.org/v14_png/temca2_3501_4200.14.0.tar"
FILENAME = "temca2_3501_4200.14.0.tar"
NUM_THREADS = 16

def get_file_size(url):
    r = requests.head(url, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers['Content-Length'])

def download_chunk(start, end, url, part_num, pbar):
    headers = {'Range': f'bytes={start}-{end}'}
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()
    with open(f"{FILENAME}.part{part_num}", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

def merge_parts(filename, num_parts):
    with open(filename, "wb") as outfile:
        for i in tqdm(range(num_parts)):
            part_file = f"{filename}.part{i}"
            with open(part_file, "rb") as infile:
                outfile.write(infile.read())
            os.remove(part_file)

def download_file(url, filename, num_threads):
    file_size = get_file_size(url)
    part_size = file_size // num_threads

    print(f"Total size: {file_size / (1024**3):.2f} GB")

    with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename) as pbar:
        futures = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(num_threads):
                start = i * part_size
                end = (file_size - 1) if i == num_threads - 1 else (start + part_size - 1)
                futures.append(executor.submit(download_chunk, start, end, url, i, pbar))

        for future in futures:
            future.result()

    merge_parts(filename, num_threads)
    print("✅ Download completed and file merged.")

if __name__ == "__main__":
    # download_file(URL, FILENAME, NUM_THREADS)
    merge_parts(FILENAME, NUM_THREADS)