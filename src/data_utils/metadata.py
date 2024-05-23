# %%
import os
from functools import cached_property
import requests
import pandas as pd
import zipfile
import pdb

def is_nan(element):
    return element != element

class PokemonMetaData():
    def __init__(self, types_path="data/types.csv"):
        self.types = self.__load_types(types_path)
        self.__add_real_types()


    def __download_types_csv(self, save_folder="data"):
        if os.path.exists(f"{save_folder}/types.csv"):
            print("types.csv already exists, skipping download")
            return
        print("Downloading types.csv")
        url = "https://gist.github.com/armgilles/194bcff35001e7eb53a2a8b441e8b2c6/archive/92200bc0a673d5ce2110aaad4544ed6c4010f687.zip"

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Download zip and extract types.csv
        response = requests.get(url)
        with open(f"{save_folder}/types.zip", 'wb') as f:
            f.write(response.content)


        with zipfile.ZipFile(f"{save_folder}/types.zip", 'r') as zip_ref:
            zip_ref.extractall(save_folder)

        # Move file out of folder
        os.rename(f"{save_folder}/194bcff35001e7eb53a2a8b441e8b2c6-92200bc0a673d5ce2110aaad4544ed6c4010f687/pokemon.csv", f"{save_folder}/types.csv")
        
        # Clean up
        os.remove(f"{save_folder}/types.zip")
        os.rmdir(f"{save_folder}/194bcff35001e7eb53a2a8b441e8b2c6-92200bc0a673d5ce2110aaad4544ed6c4010f687")

        return pd.read_csv(f"{save_folder}/types.csv")


    def __load_types(self, types_path="data/types.csv"):
        import pandas as pd

        if not os.path.exists(types_path):
            print(f"types.csv not found at {types_path}")

            types_folder = os.path.dirname(types_path)
            types = self.__download_types_csv(save_folder=types_folder)
            return types
        else:
            types = pd.read_csv(types_path)
            return types

    def __add_real_types(self):
        # apply get_type_by_id to all rows
        self.types["Real Type"] = self.types.apply(lambda row: self.get_type_by_id(row["#"]), axis=1)

    def get_type_by_id(self, id, numeric=False):
        """Gets the main type for a pokemon by id. 
        This will return type 1 for any pokemon and type 2 if type 1 is Normal
        
        Args:
            id (int): Pokemon ID
        
        Returns:
            str: Type of the pokemon
        """
        row =  self.types[self.types["#"] == id]
        # get Type1 
        type1 = row["Type 1"].values[0]

        if type1 == "Normal":
            type2 = row["Type 2"].values[0]
            if not is_nan(type2):
                return type2 if not numeric else self.types_dict[type2]
        return type1 if not numeric else self.types_dict[type1]
    
    @cached_property
    def unique_types(self):
        return self.types["Real Type"].unique()
    
    @cached_property
    def types_dict(self):
        # sort unique 
        unique_types = sorted(self.unique_types)
        return {type_: idx for idx, type_ in enumerate(unique_types)}
# %%
if __name__ == "__main__":
    metadata = PokemonMetaData("../../data/types.csv")
    types_df = metadata.types

    # check for nam in Real Types
    print(types_df["Real Type"].unique())

    print(metadata.types_dict)

# %%
