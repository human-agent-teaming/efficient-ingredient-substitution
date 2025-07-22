# load Gismo preprocessed Vocabulary

import os
from collections import defaultdict

from pathlib import Path

#TODO: First, preprocess Recipe1MSubs using flavorgraph!

ingredient_directory = "ingredient_vocabularies_and_embeddings"
Path(ingredient_directory).mkdir(parents=True, exist_ok=True)


# TODO: move the preprocessed vocabulary file under the ingredient_directory 'ingredient_vocabularies_and_embeddings'

# TODO: download and set gismo path. ..
inv_cooking_module_path = "../gismo/inv_cooking/__init__.py"


# load the "inv_cooking" module of gismo
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("inv_cooking", inv_cooking_module_path)
foo = importlib.util.module_from_spec(spec)
sys.modules["inv_cooking"] = foo
spec.loader.exec_module(foo)

# Download the trained flavorgraph embeddings from "https://github.com/lamypark/FlavorGraph" or "https://drive.google.com/file/d/1MN2dGr-e8x09XSfj0kG4MahTRFY8GDw4/view"
flavorgraph_embeddings_url = "https://drive.google.com/file/d/1MN2dGr-e8x09XSfj0kG4MahTRFY8GDw4/view?usp=sharing"
import urllib.request

Path(ingredient_directory + "/FlavorGraph").mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve("flavorgraph_embeddings_url", "ingredient_vocabularies_and_embeddings/FlavorGraph/FlavorGraph Node Embedding.pickle")

# loading GISMo's vocabulary
# TODO: Fix path and Specify the ingredient_directory of the preprocessed GISMO vocabulary
gismo_vocabulary_path = "recipe1m/preprocessed_flavorgraph_substitutions_fixed_3/final_recipe1m_vocab_ingrs.pkl"

# load the Vocabulary instance in the pickle
import pickle
with (open(gismo_vocabulary_path, "rb")) as openfile:
    recipe1MSubs_preprocessed_with_flavorgraph_vocab = pickle.load(openfile) #of type "inv_cooking.Vocabulary", for more info, see https://github.com/facebookresearch/gismo/blob/main/inv_cooking/datasets/vocabulary.py



# load flavorgraph ingredient embeddings
import pickle
# load Recipe1MSubs val split file
with open(os.path.join(ingredient_directory, "FlavorGraph Node Embedding.pickle"), "rb") as openfile:
    flavorgraph_node_embeddings = pickle.load(openfile)


print("Total GISMo ingredient vocabulary size (including '<end>' and '<pad>'):", len(recipe1MSubs_preprocessed_with_flavorgraph_vocab.word2idx))
print("Total GISMo unique ingredients (including '<end>' and '<pad>'):", len(recipe1MSubs_preprocessed_with_flavorgraph_vocab.idx2word))

# we create a clean mapping that does not include '<pad>' nor '<end>'
gismo_ingredient_name_to_new_id: dict[str, int] = dict()
new_ingredient_id_to_gismo_name: dict[int, list[str]] = dict()

for ingredient_name in recipe1MSubs_preprocessed_with_flavorgraph_vocab.word2idx:
    if ingredient_name == '<pad>' or ingredient_name == '<end>':
        continue
    gismo_ingredient_index = recipe1MSubs_preprocessed_with_flavorgraph_vocab.word2idx[ingredient_name]
    # we know that '<end>' is at index 0, and '<pad>' has the last index, so we reduce ingredient index by 1 to all remaining ingredients
    new_igredient_index = gismo_ingredient_index - 1
    gismo_ingredient_name_to_new_id[ingredient_name] = new_igredient_index
    if new_igredient_index not in new_ingredient_id_to_gismo_name:
        new_ingredient_id_to_gismo_name[new_igredient_index] = []
    new_ingredient_id_to_gismo_name[new_igredient_index].append(ingredient_name)


print("Size of new ingredient vocabulary (after removing GISMo's '<end>' and '<pad>'):", len(gismo_ingredient_name_to_new_id))
print("Number of unique ingredients:", len(new_ingredient_id_to_gismo_name))

# run a quick check that all ignredient indicies have reduced by 1.
for ingredient_name in gismo_ingredient_name_to_new_id:
    assert gismo_ingredient_name_to_new_id[ingredient_name] == recipe1MSubs_preprocessed_with_flavorgraph_vocab.word2idx[ingredient_name] - 1

# load flavorgraph node vocabulary
import csv

flavorgraph_node_id_to_name = dict()
flavorgraph_node_name_to_id = dict()

with open(os.path.join(ingredient_directory, 'FlavorGraph-master/input/nodes_191120.csv')) as csvfile: #, newline=''

    flavorgraph_nodes = csv.reader(csvfile, delimiter=',')#, quotechar='|'

    for row in flavorgraph_nodes:
        # we only keep the ingredients
        if row[3] != 'ingredient':
            continue

        node_id = int(row[0])
        node_name = row[1]
        flavorgraph_node_id_to_name[node_id] = node_name
        flavorgraph_node_name_to_id[node_name] = node_id



# gismo_ingredients_not_found_set = set()

flavorgraph_synonyms_ingredient_ids_to_be_ignored = set()

flavorgraph_ingredient_id_to_new_ingredient_id: dict[int, int] = dict()
new_ingredient_id_to_flavorgraph_id: dict[int, int] = dict()

# for each ingredient name in the gismo vocabulary
for ingredient_name in gismo_ingredient_name_to_new_id:
    new_ingredient_id = gismo_ingredient_name_to_new_id[ingredient_name]
    if ingredient_name in flavorgraph_node_name_to_id:
        flavorgraph_id = flavorgraph_node_name_to_id[ingredient_name]
        flavorgraph_id_to_keep = None


        # in case an ingredient synonym, according to Recipe1MSubs preprocessing, is matched with different ingredients in Flavorgraph:
        if new_ingredient_id in new_ingredient_id_to_flavorgraph_id and new_ingredient_id_to_flavorgraph_id[new_ingredient_id] != flavorgraph_id:
            previous_flavorgraph_ingredient_id = new_ingredient_id_to_flavorgraph_id[new_ingredient_id]
            previous_flavorgraph_ingredient_name = flavorgraph_node_id_to_name[previous_flavorgraph_ingredient_id]
            current_flavorgraph_id = flavorgraph_id
            current_flavorgraph_name = ingredient_name

            # deal with plurals:
            if previous_flavorgraph_ingredient_name[:-2] in current_flavorgraph_name[:-2]:
                flavorgraph_id_to_keep = previous_flavorgraph_ingredient_id

                flavorgraph_synonyms_ingredient_ids_to_be_ignored.add(current_flavorgraph_id)
            elif current_flavorgraph_name[:-2] in previous_flavorgraph_ingredient_name[:-2]:
                flavorgraph_id_to_keep = current_flavorgraph_id
                flavorgraph_synonyms_ingredient_ids_to_be_ignored.add(previous_flavorgraph_ingredient_id)
            # deal with this special case of synonyms
            elif current_flavorgraph_name == 'pimiento' or current_flavorgraph_name == 'pimento' or current_flavorgraph_name == 'pimento_pepper':
                flavorgraph_id_to_keep = flavorgraph_node_name_to_id['pimiento']
            # else:
            #     print(f"gismo ID: {gismo_ingredient_id} found already.\nPrevious flavorgraph ID: {new_ingredient_id_to_flavorgraph_id[gismo_ingredient_id]} and current flavorgraph Id: {flavorgraph_id}\nOld ingredient name: {flavorgraph_node_id_to_name[new_ingredient_id_to_flavorgraph_id[gismo_ingredient_id]]}, new name: {flavorgraph_node_id_to_name[flavorgraph_id]}.\n")
        else:
            flavorgraph_id_to_keep = flavorgraph_id

        assert flavorgraph_id_to_keep is not None

        flavorgraph_ingredient_id_to_new_ingredient_id[flavorgraph_id_to_keep] = new_ingredient_id
        new_ingredient_id_to_flavorgraph_id[new_ingredient_id] = flavorgraph_id_to_keep

    # else:
    #     gismo_ingredients_not_found_set.add(ingredient_name)



flavorgraph_synonyms_ingredient_ids_to_be_ignored.add(flavorgraph_node_name_to_id['pimento'])
flavorgraph_synonyms_ingredient_ids_to_be_ignored.add(flavorgraph_node_name_to_id['pimento_pepper'])

print(f"Found {len(flavorgraph_synonyms_ingredient_ids_to_be_ignored)} ingredient names, that are synonyms according to GISMo, but different according to Flavorgraph.\nThey are synonyms, plurals, etc. and one of each is selected as correct mapping.")
for flavorgraph_ingredient_id_to_be_removed in flavorgraph_synonyms_ingredient_ids_to_be_ignored:
    if flavorgraph_ingredient_id_to_be_removed in flavorgraph_ingredient_id_to_new_ingredient_id:
        del flavorgraph_ingredient_id_to_new_ingredient_id[flavorgraph_ingredient_id_to_be_removed]

print("Number of FlavorGraph ingredients:", len(flavorgraph_ingredient_id_to_new_ingredient_id))
with open(os.path.join(ingredient_directory, 'flavorgraph_ingredient_id_to_new_ingredient_id.pickle'), 'wb') as handle:
    pickle.dump(flavorgraph_ingredient_id_to_new_ingredient_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Number of unique ingredients:", len(new_ingredient_id_to_flavorgraph_id))
with open(os.path.join(ingredient_directory, 'new_ingredient_id_to_flavorgraph_id.pickle'), 'wb') as handle:
    pickle.dump(new_ingredient_id_to_flavorgraph_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

# also write to a file the gismo gismo_ingredient_id_to_gismo_ingredient_name and gismo_ingredient_name_to_gismo_ingredient_id mappings.
print("Number of GISMo Ingredient names:", len(gismo_ingredient_name_to_new_id))
with open(os.path.join(ingredient_directory,
                       'gismo_ingredient_name_to_new_id.pickle'),
          'wb') as handle:
    pickle.dump(gismo_ingredient_name_to_new_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Number of new Ingredient IDs:", len(new_ingredient_id_to_gismo_name))
with open(os.path.join(ingredient_directory,
                       'new_ingredient_id_to_gismo_name.pickle'),
          'wb') as handle:
    pickle.dump(new_ingredient_id_to_gismo_name, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Total gismo ingredient names (including '<end>' and '<pad>'): 10131
# Found 20 ingredient names, that are synonyms according to GISMo, but different according to Flavorgraph.
# They are synonyms, plurals, etc. and one of each is selected as correct mapping.
# Number of gismo ingredient names (synonyms are treated separately): 10129
# Number of FlavorGraph ingredients: 6632
# Number of unique gismo ingredients: 6632
# Number of flavorgraph ingredients used: 6632
# Number of GISMo Ingredient names (including '<end>' and '<pad>'): 10131
# Number of GISMo Ingredient IDs (including '<end>' and '<pad>'): 6634

# Total GISMo ingredient vocabulary size (including '<end>' and '<pad>'): 10131
# Total GISMo unique ingredients (including '<end>' and '<pad>'): 6634
# Size of new ingredient vocabulary (after removing GISMo's '<end>' and '<pad>'): 10129
# Number of unique ingredients: 6632
# Found 20 ingredient names, that are synonyms according to GISMo, but different according to Flavorgraph.
# They are synonyms, plurals, etc. and one of each is selected as correct mapping.
# Number of FlavorGraph ingredients: 6632
# Number of unique ingredients: 6632
# Number of GISMo Ingredient names: 10129
# Number of new Ingredient IDs: 6632