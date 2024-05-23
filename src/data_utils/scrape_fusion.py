# %%
import os
import requests

import pandas as pd
import zipfile
import itertools

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data_utils.metadata import PokemonMetaData
# %%
def get_img_urls():
    url = "https://fusioncalc.com/"

    # Set up Selenium options
    options = Options()
    options.add_argument("--headless")  # Run in headless mode (no browser UI)
    options.add_argument("--disable-gpu")  # Disable GPU rendering for better compatibility
    options.add_argument("--no-sandbox")  # Bypass OS security model
    options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems

    # Set up the WebDriver
    driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)
    driver.get(url)

    img_element1 = driver.find_element(By.ID, "pic1")
    img_element2 = driver.find_element(By.ID, "pic2")

    image_url = img_element1.get_attribute('src')
    image_url2 = img_element2.get_attribute('src')

    driver.quit()
    return image_url, image_url2


def get_fusion(id1, id2, save_folder="data/fusion"):

    fusion_url = f"https://fusioncalc.com/wp-content/themes/twentytwentyone/pokemon/autogen-fusion-sprites-master/Battlers/{id1}/{id1}.{id2}.png"

    # Download the image
    response = requests.get(fusion_url)

    # create the save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if response.status_code == 200:
        # print(f"Downloading image: {fusion_url}")
        with open(f"{save_folder}/fusion_{id1}_{id2}.png", 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download image: {fusion_url}")


# %%
if __name__ == "__main__":
    metadata = PokemonMetaData("./data/types.csv")
    types_df = metadata.types

    # %% 

    # Only get the first two generations
    types_gen_df = types_df[types_df["Generation"] <= 2]

    # Function to get combinations of IDs within each type
    def get_combinations(df):
        result = []
        for type_, group in df.groupby('Real Type'):
            ids = group['#'].tolist()
            combinations = list(itertools.product(ids, repeat=2))
            for comb in combinations:
                result.append((type_, comb[0], comb[1]))
        return pd.DataFrame(result, columns=['type', 'id1', 'id2'])

    # Get the combinations
    combinations_df = get_combinations(types_gen_df)


    # %%
        
    # Function to process a row in parallel
    def process_row(row):
        type_, id1, id2 = row.type, row.id1, row.id2
        result = get_fusion(id1, id2, save_folder=f"./data/fusion/{type_}")
        return result

    # Run the fusion generation in parallel with a progress bar
    def run_parallel_fusion(combinations_df):
        results = []
        with ThreadPoolExecutor(max_workers=16) as executor:  # Adjust the number of workers as needed
            futures = {executor.submit(process_row, row): row for row in combinations_df.itertuples(index=False)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Rows"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing row: {e}")
        return results

    # Execute the parallel processing
    fusion_results = run_parallel_fusion(combinations_df)

# %%
