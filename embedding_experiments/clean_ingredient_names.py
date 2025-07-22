import pickle5 as pickle
from utils.vocabulary import IngredientsCleaner
from typing import Dict


with open("vocabulary/gismo_ingredient_name_to_flavorgraph_ingredient_name_dict.pickle", "rb") as file:
    food_to_embeddings_dict = pickle.load(file)

mapping_gismo_names: Dict[str, str] = dict()
cleaned_names: Dict[str, str] = dict()

for key, value in food_to_embeddings_dict.items():
    cleaned_key = IngredientsCleaner().clean_ingredients_names(key, use_underscores=True, remove_plurals=True)
    cleaned_value = IngredientsCleaner().clean_ingredients_names(value, use_underscores=True, remove_plurals=True)
    cleaned_names[cleaned_key] = cleaned_value
    mapping_gismo_names[key] = cleaned_key

with open("vocabulary/gismo_ingredient_name_to_flavorgraph_ingredient_name_dict_cleaned_new_version.pickle", "wb") as file:
    pickle.dump(cleaned_names, file)
with open("vocabulary/gismo_names_mapping.pickle", "wb") as file:
    pickle.dump(mapping_gismo_names, file)