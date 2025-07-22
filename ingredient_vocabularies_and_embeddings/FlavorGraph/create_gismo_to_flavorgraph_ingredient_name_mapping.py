# import gismo Vocabulary class


# load the "inv_cooking" module of gismo !
import importlib.util
import sys

# TODO: make sure GISMo is downloaded
spec = importlib.util.spec_from_file_location("inv_cooking",
                                              "gismo/inv_cooking/__init__.py")
foo = importlib.util.module_from_spec(spec)
sys.modules["inv_cooking"] = foo
spec.loader.exec_module(foo)

# load the Vocabulary instance in the pickle
import pickle
# TODO: Fix path
with (
open("recipe1m/preprocessed_flavorgraph_substitutions_fixed_3/final_recipe1m_vocab_ingrs.pkl",
     "rb")) as openfile:
    gismo_preprocessed_with_flevorgraph_recipe1m_vocab = pickle.load(openfile)


# import flavorgraph ingredient embeddings

import pickle

# TODO: Fix path
# load Recipe1MSubs val split file
with (
open("/ingredient-substitution/ingredient_vocabularies_and_embeddings/FlavorGraph/FlavorGraph Node Embedding.pickle",
     "rb")) as openfile:
    flavorgraph_node_embeddings = pickle.load(openfile)

# read flavorgraph node vocabulary

import csv

flavorgraph_node_id_to_name = dict()
flavorgraph_node_name_to_id = dict()

# TODO: Fix path
with open(
        'ingredient-substitution/ingredient_vocabularies_and_embeddings/FlavorGraph/nodes_191120.csv') as csvfile:  # , newline=''

    flavorgraph_nodes = csv.reader(csvfile, delimiter=',')  # , quotechar='|'

    for row in flavorgraph_nodes:
        # we only keep the ingredients
        if row[3] != 'ingredient':
            continue

        node_id = int(row[0])
        node_name = row[1]
        flavorgraph_node_id_to_name[node_id] = node_name
        flavorgraph_node_name_to_id[node_name] = node_id

print(len(flavorgraph_node_name_to_id))

# load new ingredient id to flavorgraph id

# TODO: Fix path
with (
open("ingredient-substitution/ingredient_vocabularies_and_embeddings/new_ingredient_id_to_flavorgraph_id.pickle",
     "rb")) as openfile:
    new_ingredient_id_to_flavorgraph_id = pickle.load(openfile)

ingredient_names_list = []

for new_id in new_ingredient_id_to_flavorgraph_id:
    flavorgraph_id = new_ingredient_id_to_flavorgraph_id[new_id]
    flavorgraph_name = flavorgraph_node_id_to_name[flavorgraph_id]
    ingredient_names_list.append(flavorgraph_name)

# TODO: Fix path
with open(
        "ingredient-substitution/ingredient_vocabularies_and_embeddings/flavorgraph_ingredinets_used_name_list.pickle",
        'wb') as handle:
    pickle.dump(ingredient_names_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ['node_id', 'name', 'id', 'node_type', 'is_hub']
# ['0', '1%_fat_buttermilk', '', 'ingredient', 'no_hub']
# ['1', '1%_fat_cottage_cheese', '', 'ingredient', 'no_hub']
# ['3', '10%_cream', '', 'ingredient', 'no_hub']
# ['4', '100%_bran', '', 'ingredient', 'no_hub']
# ['5', '10_inch_flour_tortilla', '', 'ingredient', 'no_hub']
# ['7', '12_inch_pizza_crust', '', 'ingredient', 'no_hub']
# ['9', '18%_table_cream', '', 'ingredient', 'no_hub']
# ['10', '2%_buttermilk', '', 'ingredient', 'no_hub']
# ['11', '2%_cheddar_cheese', '', 'ingredient', 'no_hub']
# ['12', '2%_evaporated_milk', '', 'ingredient', 'no_hub']


# let's see how many ingredients we can match between flavorgraph and recipe1MSubs
ingredients_not_found_counter = 0

ingredient_ids_not_found_set = set()

ingredient_found_counter = 0

ingredient_ids_found_set = set()

gismo_ingredient_id_to_flavorgraph_id = dict()

print(len(gismo_preprocessed_with_flevorgraph_recipe1m_vocab.word2idx))

for ingredient_name in gismo_preprocessed_with_flevorgraph_recipe1m_vocab.word2idx:
    gismo_ingredient_id = gismo_preprocessed_with_flevorgraph_recipe1m_vocab.word2idx[ingredient_name]
    if ingredient_name in flavorgraph_node_name_to_id:
        flavorgraph_id = flavorgraph_node_name_to_id[ingredient_name]
        flavorgraph_id_to_keep = None
        if gismo_ingredient_id in gismo_ingredient_id_to_flavorgraph_id and gismo_ingredient_id_to_flavorgraph_id[
            gismo_ingredient_id] != flavorgraph_id:
            previous_flavorgraph_ingredient_id = gismo_ingredient_id_to_flavorgraph_id[gismo_ingredient_id]
            previous_flavorgraph_ingredient_name = flavorgraph_node_id_to_name[previous_flavorgraph_ingredient_id]
            current_flavorgraph_id = flavorgraph_id
            current_flavorgraph_name = flavorgraph_node_id_to_name[current_flavorgraph_id]

            # deal with plurals:
            if previous_flavorgraph_ingredient_name[:-2] in current_flavorgraph_name[:-2]:
                flavorgraph_id_to_keep = previous_flavorgraph_ingredient_id
            elif current_flavorgraph_name[:-2] in previous_flavorgraph_ingredient_name[:-2]:
                flavorgraph_id_to_keep = current_flavorgraph_id
            elif current_flavorgraph_name == 'pimiento' or current_flavorgraph_name == 'pimento' or current_flavorgraph_name == 'pimento_pepper':
                flavorgraph_id_to_keep = flavorgraph_node_name_to_id['pimiento']
        else:
            flavorgraph_id_to_keep = flavorgraph_id

        gismo_ingredient_id_to_flavorgraph_id[gismo_ingredient_id] = flavorgraph_id_to_keep
        ingredient_found_counter += 1
        ingredient_ids_found_set.add(gismo_ingredient_id)

        assert flavorgraph_id_to_keep is not None
    else:
        ingredients_not_found_counter += 1
        ingredient_ids_not_found_set.add(gismo_ingredient_id)

print("FlavorGraph ingredient vocabulary size:", len(flavorgraph_node_id_to_name))
ingredient_ids_not_found = ingredient_ids_not_found_set - ingredient_ids_found_set

print("ingredients found:", ingredient_found_counter)
print("Total ingredient id vocabulary found:", len(ingredient_ids_found_set))
print("ingredients NOT found:", ingredients_not_found_counter)
print("Total ingredient id vocabulary NOT found:", len(ingredient_ids_not_found))
print("Ingredient IDs not found:", ingredient_ids_not_found)


import os


gismo_ingredient_name_to_flavorgraph_ingredient_name = dict()

for gismo_ingredient_name in gismo_preprocessed_with_flevorgraph_recipe1m_vocab.word2idx:
    if gismo_ingredient_name != "<end>" and gismo_ingredient_name != "<pad>":
        gismo_ingredient_id = gismo_preprocessed_with_flevorgraph_recipe1m_vocab.word2idx[gismo_ingredient_name]
        flavorgraph_ingredient_id = gismo_ingredient_id_to_flavorgraph_id[gismo_ingredient_id]
        flavorgraph_ingredient_name = flavorgraph_node_id_to_name[flavorgraph_ingredient_id]

        gismo_ingredient_name_to_flavorgraph_ingredient_name[gismo_ingredient_name] = flavorgraph_ingredient_name

print("total gismo ingredient names:", len(gismo_ingredient_name_to_flavorgraph_ingredient_name))
print("total flavorgraph ingredient names:", len(set(gismo_ingredient_name_to_flavorgraph_ingredient_name.values())))

# TODO: Fix path
with open(os.path.join(
        "ingredient-substitution/ingredient_vocabularies_and_embeddings",
        'gismo_ingredient_name_to_flavorgraph_ingredient_name_dict.pickle'), 'wb') as handle:
    pickle.dump(gismo_ingredient_name_to_flavorgraph_ingredient_name, handle)

# total gismo ingredient names: 10129
# total flavorgraph ingredient names: 6632

