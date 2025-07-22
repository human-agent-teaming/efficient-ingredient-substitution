import pickle
from collections import defaultdict

with (
open("ingredient_vocabularies_and_embeddings/ontology_properties/new_ingredient_id_to_food_on_matches_min_conf_0.6.pickle",
     "rb")) as openfile:
    food_on_matches = pickle.load(openfile)

# import foodOn complete

ontologies_dir:str = "ingredient_vocabularies_and_embeddings/ontology_properties/foodon_ontologies/"

import rdflib
from rdflib import RDFS

foodon_graph = rdflib.Graph()
foodon_graph.parse(ontologies_dir + 'foodon_complete.owl', format='xml')

from rdflib import RDFS, RDF, URIRef, OWL

from collections import defaultdict

from rdflib import Graph, URIRef, RDFS


def get_superclasses(foodon_graph, start_class):
    # Initialize a dictionary to store superclasses and their distances
    superclasses = {}

    def recurse(class_uri, distance):
        for superclass in foodon_graph.objects(class_uri, RDFS.subClassOf):
            if superclass not in superclasses:
                superclasses[superclass] = distance
                recurse(superclass, distance + 1)

    # Start the recursion with the initial class and distance 1
    recurse(start_class, 2)
    return superclasses

ignore_blank_nodes: bool = True

for max_distance in [5]:
    # for max_distance in [5]:
    print("Max distance:", max_distance)

    ingredient_id_to_food_on_property_and_distance_dict: dict[int, dict[URIRef, int]] = defaultdict(dict)
    # for ingredient_id in food_on_matches:
    #     ingredient_id_to_food_on_property_and_distance_dict[ingredient_id] = {}

    foodon_classes_counter = defaultdict(int)

    foodOn_complete_vocabulary: set[URIRef] = set()

    # RDF.type to get the class of an entity
    # flag = False
    for ingredient_id in food_on_matches:
        for food_on_entity_match in food_on_matches[ingredient_id]:

            if food_on_entity_match not in ingredient_id_to_food_on_property_and_distance_dict[ingredient_id]:
                foodon_classes_counter[food_on_entity_match] += 1
            # set this property as distance equal to 1
            ingredient_id_to_food_on_property_and_distance_dict[ingredient_id][food_on_entity_match] = 1

            foodOn_complete_vocabulary.add(food_on_entity_match)

            superclasses = get_superclasses(foodon_graph, food_on_entity_match)

            # Print the superclasses and their distances
            for superclass, distance in superclasses.items():

                # skip "empty" class nodes
                if ignore_blank_nodes and str(superclass).startswith("N"):
                    continue

                if superclass == OWL.Thing:
                    continue

                if distance > max_distance:
                    continue

                foodOn_complete_vocabulary.add(superclass)

                if superclass not in ingredient_id_to_food_on_property_and_distance_dict[ingredient_id]:
                    ingredient_id_to_food_on_property_and_distance_dict[ingredient_id][superclass] = distance
                    foodon_classes_counter[superclass] += 1
                else:
                    # we keep the maximum distance found between our ingredient and a class.
                    # to prioritize more classes with smaller distances (that are probably more specific classes)
                    current_distance = ingredient_id_to_food_on_property_and_distance_dict[ingredient_id][superclass]
                    if current_distance > distance:
                        ingredient_id_to_food_on_property_and_distance_dict[ingredient_id][superclass] = distance

    print("Total vocabulary of FoodOn properties retrieved:", len(foodOn_complete_vocabulary))

    import numpy as np
    import numpy.typing as npt

    index_to_food_on_class: dict[int, URIRef] = dict()
    food_on_class_to_index: dict[URIRef, int] = dict()

    food_on_class_will_be_used: dict[URIRef, bool] = dict()

    food_on_index: int = 0

    total_ingredients_matched_with_food_on: int = len(food_on_matches)

    for food_on_class in foodon_classes_counter:
        if foodon_classes_counter[food_on_class] > 1 and foodon_classes_counter[
            food_on_class] != total_ingredients_matched_with_food_on:

            food_on_class_will_be_used[food_on_class] = True

            index_to_food_on_class[food_on_index] = food_on_class
            food_on_class_to_index[food_on_class] = food_on_index
            food_on_index += 1
        else:
            food_on_class_will_be_used[food_on_class] = False

    print(f"A total of {food_on_index} food On categories will be used.")

    flavorgraph_ing_vocab_size: int = 6632

    food_on_categorical_embeddings: npt.NDArray[np.float32] = np.zeros((flavorgraph_ing_vocab_size, food_on_index),
                                                                       dtype=np.float32)


    # delete foodOn properties that will not be used
    for ingredient_id in ingredient_id_to_food_on_property_and_distance_dict:
        foodOn_classes_to_delete: set[URIRef] = set()
        for food_on_class in ingredient_id_to_food_on_property_and_distance_dict[ingredient_id]:
            if not food_on_class_will_be_used[food_on_class]:
                foodOn_classes_to_delete.add(food_on_class)
        for food_on_class_to_delete in foodOn_classes_to_delete:
            del ingredient_id_to_food_on_property_and_distance_dict[ingredient_id][food_on_class_to_delete]

    food_on_class_to_label: dict[URIRef, str] = dict()
    rdfs_labels_found: int = 0

    # retrieve foodOn class Labels for the used classes

    for foodOn_URIRef in food_on_class_will_be_used:
        if food_on_class_will_be_used[foodOn_URIRef]:
            foodon_rdfs_label = next(foodon_graph.objects(subject=URIRef(foodOn_URIRef), predicate=RDFS.label), None)
            food_on_class_to_label[foodOn_URIRef] = foodon_rdfs_label
            if foodon_rdfs_label is not None:
                rdfs_labels_found += 1

    # print("rdfs:Label(s) found:", rdfs_labels_found)

    for ingredient_id in ingredient_id_to_food_on_property_and_distance_dict:
        for food_on_class in ingredient_id_to_food_on_property_and_distance_dict[ingredient_id]:

            if food_on_class_will_be_used[food_on_class]:
                distance = ingredient_id_to_food_on_property_and_distance_dict[ingredient_id][food_on_class]
                #
                embedding_value = 2 ** (- (distance + 1))
                #
                food_on_class_index = food_on_class_to_index[food_on_class]

                food_on_categorical_embeddings[ingredient_id, food_on_class_index] = embedding_value

    with (
    open(f"ingredient_vocabularies_and_embeddings/ontology_properties/new_ingredient_id_to_foodOn_properties_embeddings_{max_distance}_hops_quadr_values_WO_blank_nodes.pickle",
         "wb")) as openfile:
        pickle.dump(food_on_categorical_embeddings, openfile)

    with (
    open(f"ingredient_vocabularies_and_embeddings/ontology_properties/new_ingredient_id_to_foodOn_properties_and_distances_{max_distance}_WO_blank_nodes.pickle",
         "wb")) as openfile:
        pickle.dump(ingredient_id_to_food_on_property_and_distance_dict, openfile)

    with (
    open(f"ingredient_vocabularies_and_embeddings/ontology_properties/foodOn_index_to_URIRef_{max_distance}_WO_blank_nodes.pickle",
         "wb")) as openfile:
        pickle.dump(index_to_food_on_class, openfile)

    with (
    open(f"ingredient_vocabularies_and_embeddings/ontology_properties/foodOn_URIRef_to_foodOn_index_{max_distance}_WO_blank_nodes.pickle",
         "wb")) as openfile:
        pickle.dump(food_on_class_to_index, openfile)

    with open(
            f"ingredient_vocabularies_and_embeddings/ontology_properties/FoodOn_class_to_Label_{max_distance}_WO_blank_nodes.pickle",
            'wb') as handle:
        pickle.dump(food_on_class_to_label, handle)


# Max distance: 5
# Total vocabulary of FoodOn properties: 4022
# A total of 2368 food On categories will be used.