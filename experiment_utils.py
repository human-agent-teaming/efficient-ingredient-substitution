import pickle, os
from collections import defaultdict
from typing import Union, Optional
import statistics
import re


def create_hashable_example_input(unhashable_example_input: tuple[str, set[int], int]) -> tuple[str, frozenset[int], int]:
    recipe_id:str = unhashable_example_input[0]
    recipe_ingredients:set[int] = unhashable_example_input[1]
    source_ingredient:int = unhashable_example_input[2]
    hashable_example_input = (recipe_id, frozenset(recipe_ingredients), source_ingredient)
    return hashable_example_input

# Load Recipe1MSubs data
def load_recipe_1MSubs_dataset(recipe1Msubs_path) -> dict[str, list]:
    recipe_subs_data: dict[str, list] = dict()

    # load Recipe1MSubs train split file
    with (open(os.path.join(recipe1Msubs_path, "train_comments_subs.pkl"), "rb")) as openfile:
        recipe_subs_data["train"] = pickle.load(openfile)

    # load Recipe1MSubs val split file
    with (open(os.path.join(recipe1Msubs_path, "val_comments_subs.pkl"), "rb")) as openfile:
        recipe_subs_data["val"] = pickle.load(openfile)

    # load Recipe1MSubs test split file
    with (open(os.path.join(recipe1Msubs_path, "test_comments_subs.pkl"), "rb")) as openfile:
        recipe_subs_data["test"] = pickle.load(openfile)

    return recipe_subs_data

def translate_recipe1MSubs_data_using_gismo_ingredient_IDs(recipe_subs_data: dict[str, list], ingredient_name_to_index: dict[str, int]) -> dict[str, dict[str, list[Union[int, tuple[str, set[int], int]]]]]:
    translated_recipe_subs_data: dict[str, dict[str, list[Union[int, tuple[str, set[int], int]]]]] = dict()

    ingredient_vocabulary_used:set = set()

    recipe_ingredient_not_found_counter = 0
    source_ingredient_not_found_counter = 0
    target_ingredient_not_found_counter = 0

    recipe_ingredients_missed_vocab = set()
    source_ingredients_missed_vocab = set()
    target_ingredients_missed_vocab = set()

    substitution_samples_missed = defaultdict(int)

    for split in recipe_subs_data:
        split_example_inputs: list[tuple[str, set[int], int]] = []
        split_targets: list[int] = []
        translated_recipe_subs_data[split] = {"inputs": split_example_inputs, "targets": split_targets}

        for substitution_entry in recipe_subs_data[split]:
            use_recipe_flag = True

            recipe_id: str = substitution_entry["id"]
            recipe_ingredients: list[list[str]] = substitution_entry["ingredients"]
            suggested_substitution: tuple[str, str] = substitution_entry["subs"]
            source_ingredient: str = suggested_substitution[0]
            target_ingredient: str = suggested_substitution[1]

            translated_recipe_ingredients = set()
            for ingredient_synonyms_list in recipe_ingredients:
                ingredient_found_flag = False
                for ingredient_synonym in ingredient_synonyms_list:
                    if ingredient_synonym in ingredient_name_to_index:
                        ingredient_idx: int = ingredient_name_to_index[ingredient_synonym]
                        translated_recipe_ingredients.add(ingredient_idx)
                        ingredient_found_flag = True
                        ingredient_vocabulary_used.add(ingredient_synonym)
                    # only adding each ingredient synonym's idx once
                        break
                if not ingredient_found_flag:
                    recipe_ingredient_not_found_counter += 1
                    recipe_ingredients_missed_vocab.update(ingredient_synonyms_list)

            if len(translated_recipe_ingredients) < 2:
                use_recipe_flag = False

            if source_ingredient not in ingredient_name_to_index:
                source_ingredient_not_found_counter += 1
                source_ingredients_missed_vocab.add(source_ingredient)
                use_recipe_flag = False
            else:

                ingredient_vocabulary_used.add(source_ingredient)
                translated_source_ingredient:int = ingredient_name_to_index[source_ingredient]

            if target_ingredient not in ingredient_name_to_index:
                target_ingredient_not_found_counter += 1
                target_ingredients_missed_vocab.add(target_ingredient)
                use_recipe_flag = False
            else:

                ingredient_vocabulary_used.add(target_ingredient)
                translated_target_ingredient:int = ingredient_name_to_index[target_ingredient]

            if use_recipe_flag == False:
                substitution_samples_missed[split] += 1
            else:
                example_input: tuple[str, set[int], int] = (recipe_id, translated_recipe_ingredients, translated_source_ingredient)

                split_example_inputs.append(example_input)
                split_targets.append(translated_target_ingredient)

    print("Total ingredient vocabulary used:", len(ingredient_vocabulary_used), ", out of vocabulary size:", len(ingredient_name_to_index))
    print("Recipe ingredients not found:", recipe_ingredient_not_found_counter)
    print("Recipe ingredients not found (vocab size):", len(recipe_ingredients_missed_vocab))
    print("Source ingredients not found:", source_ingredient_not_found_counter)
    print("Source ingredients not found (vocab size):", len(source_ingredients_missed_vocab))
    print("Target ingredients not found:", target_ingredient_not_found_counter)
    print("Target ingredients not found (vocab size):", len(target_ingredients_missed_vocab))

    print("total ingredient vocab missed len: ", len(recipe_ingredients_missed_vocab.union(source_ingredients_missed_vocab).union(target_ingredients_missed_vocab)))
    print("Substitution samples missed:", substitution_samples_missed)

    return translated_recipe_subs_data


def evaluate_student_multithread(input):
    student, substitution_input, substitution_target = input
    return student.suggest_substitution(substitution_input).index(substitution_target)

def add_results_dictionary_values_to_overall_results_dictionary(total_evaluation_metrics_per_evaluation_step:dict[int, dict[str, list[float]]], evaluation_metrics_per_step:dict[int, dict[str, float]]) -> dict[int, dict[str, list[float]]]:
    for evaluation_step in evaluation_metrics_per_step:
        # initialize the results' dictionary if necessary
        if evaluation_step not in total_evaluation_metrics_per_evaluation_step:
            total_evaluation_metrics_per_evaluation_step[evaluation_step] = {}
        for evaluation_metric in evaluation_metrics_per_step[evaluation_step]:
            # initialize the results' list if necessary
            if evaluation_metric not in total_evaluation_metrics_per_evaluation_step[evaluation_step]:
                total_evaluation_metrics_per_evaluation_step[evaluation_step][evaluation_metric] = []
            # add the result of this step, for the evaluation metric, in the corresponding list
            total_evaluation_metrics_per_evaluation_step[evaluation_step][evaluation_metric].append(
                evaluation_metrics_per_step[evaluation_step][evaluation_metric])
    return total_evaluation_metrics_per_evaluation_step

def aggregate_experiment_repetition_results(total_evaluation_metrics_per_evaluation_step:dict[int, dict[str, list[float]]]) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    mean_eval_metrics_pers_step: dict[int, dict[str, float]] = {}
    std_eval_metrics_pers_step: dict[int, dict[str, float]] = {}
    for eval_step in total_evaluation_metrics_per_evaluation_step:
        if eval_step not in mean_eval_metrics_pers_step:
            mean_eval_metrics_pers_step[eval_step] = dict()
            std_eval_metrics_pers_step[eval_step] = dict()
        for eval_metric in total_evaluation_metrics_per_evaluation_step[eval_step]:
            recorded_evaluation_metrics_values = total_evaluation_metrics_per_evaluation_step[eval_step][eval_metric]
            mean_eval_metrics_pers_step[eval_step][eval_metric] = sum(recorded_evaluation_metrics_values) / len(recorded_evaluation_metrics_values)
            if len(total_evaluation_metrics_per_evaluation_step[eval_step][eval_metric]) == 1:
                std_eval_metrics_pers_step[eval_step][eval_metric] = 0.0
            else:
                std_eval_metrics_pers_step[eval_step][eval_metric] = statistics.stdev(recorded_evaluation_metrics_values)

    return mean_eval_metrics_pers_step, std_eval_metrics_pers_step



# def import_gismo_vocabulary_of_Vocabulary_instance(module_name, module_init_filepath, vocabulary_instance_filepath):
#     # load the "inv_cooking" module of gismo adn then the instance of the Vocabulary class
#     import importlib.util
#     import sys
#     import pickle
#     # load the gismo module "inv_cooking" from it's __init__.py file
#     spec = importlib.util.spec_from_file_location(module_name, module_init_filepath)
#     foo = importlib.util.module_from_spec(spec)
#     sys.modules[module_name] = foo
#     spec.loader.exec_module(foo)
#     # foo.Vocabulary()
#
#     # load the Vocabulary instance from the pickle file
#     with (open(vocabulary_instance_filepath, "rb")) as openfile:
#        gismo_preprocessed_with_flevorgraph_recipe1m_vocab = pickle.load(openfile)
#
#     return gismo_preprocessed_with_flevorgraph_recipe1m_vocab
