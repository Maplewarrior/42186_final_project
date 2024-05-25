import torch 
import requests
import os
from PIL import Image

def download_data(dataset_urls, data_raw_path):
    # make raw folder if it doesn't exist
    if not os.path.exists(data_raw_path):
        os.makedirs(data_raw_path)

    for gen, url in dataset_urls.items():
        # filename = os.path.join(data_raw_path, url.split('/')[-1])

        filename = f"generation-{gen}.tar.gz"

        foldername = f"generation-{gen}"
        folderpath = os.path.join(data_raw_path, foldername)
        if not os.path.exists(folderpath):
            print(f"Downloading {url} to {filename}")
            r = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(r.content)

            # extract the tar file and rename the extracted folder
            print(f"Extracting {filename}")
            os.system(f"tar -xzf {filename} -C {data_raw_path}")
            os.rename(os.path.join(data_raw_path, 'pokemon'), os.path.join(data_raw_path, foldername))

            # delete the tar file
            print(f"Deleting {filename}")
            os.remove(filename)
        else:
            print(f"File {filename} already exists, skipping download") 


def resize_image_with_white_background(filepath, new_path, size, padding_ratio=0.01):
    # Load the image
    img = Image.open(filepath).convert("RGBA")
    
    # Resize the cropped image using LANCZOS resampling for high quality
    img_resized = img.resize((size, size), Image.Resampling.LANCZOS)
    
    # Create a new white background image
    white_bg = Image.new('RGBA', (size, size), 'WHITE')

    # Remove alpha from the resized image by compositing it over a white background
    alpha_composite = Image.alpha_composite(white_bg, img_resized)
    
    # Convert back to RGB to discard alpha channel
    final_img = alpha_composite.convert("RGB")

    # Save the image with a white background
    final_img.save(new_path, 'PNG')

def process_subfolder(folderpath, new_folderpath, size):
    # Check and create new folder if it doesn't exist
    if not os.path.exists(new_folderpath):
        os.makedirs(new_folderpath)

    # Process all PNG files in the folder
    for filename in os.listdir(folderpath):
        if filename.endswith('.png'):
            filepath = os.path.join(folderpath, filename)
            new_path = os.path.join(new_folderpath, filename)
            resize_image_with_white_background(filepath, new_path, size)

def resize_images(data_raw_path, data_processed_path, size=40):
    for gen in range(1, 6):
        generation_folder = f"generation-{gen}"
        base_folder = f"{generation_folder}/main-sprites"
        folderpath = os.path.join(data_raw_path, base_folder)

        for subfolder in os.listdir(folderpath):
            subfolder_paths = {
                "front": "",
                "back": "back",
                "shiny": "shiny"
            }

            for key, value in subfolder_paths.items():
                current_path = os.path.join(folderpath, subfolder, value)
                new_folderpath = os.path.join(data_processed_path, generation_folder, subfolder, key)
                if os.path.exists(current_path):
                    process_subfolder(current_path, new_folderpath, size)

if __name__ == '__main__':


    dataset_urls = {1: "https://veekun.com/static/pokedex/downloads/generation-1.tar.gz",
                2: "https://veekun.com/static/pokedex/downloads/generation-2.tar.gz",
                3: "https://veekun.com/static/pokedex/downloads/generation-3.tar.gz",
                4: "https://veekun.com/static/pokedex/downloads/generation-4.tar.gz",
                5: "https://veekun.com/static/pokedex/downloads/generation-5.tar.gz"}


    data_raw_path = 'data/raw/'
    data_processed_path = 'data/processed/'

    download_data(dataset_urls, data_raw_path)

    resize_images(data_raw_path, data_processed_path, size=64)

