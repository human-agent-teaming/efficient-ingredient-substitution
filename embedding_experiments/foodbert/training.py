import json
import sys
from pathlib import Path
import pickle5 as pickle
from embedding_experiments.utils.vocabulary import extend_vocabulary, find_common_ingredients
from embedding_experiments.foodbert.run_language_modeling import main
import numba.cuda as cuda


def device():
    global selected_device
    devices = cuda.list_devices()
    num_devices = len(devices)
    selected_device = None
    for i in range(1, num_devices):
        if devices[i].compute_capability != (0, 0):
            selected_device = i
            break


if __name__ == "__main__":

    cuda.select_device(device())
    path_dir = Path.cwd()/'embedding_experiments'

    ################# Preparing vocabularies

    bert_path = path_dir/'foodbert/data/bert-base-cased-original.txt'
    ingredient_path = path_dir/'vocabulary/gismo_ingredient_name_to_flavorgraph_ingredient_name_dict_cleaned_new_version.pickle'
    flavor_path_json = path_dir/'foodbert/data/ingredient_names_valid.json'

    with open(ingredient_path, 'rb') as fb:
        ingredients_dictionary = pickle.load(fb)
        
    with open(bert_path, 'r') as fb:
        bert_vocabulary = fb.read()


    # Avoid ingredients already present in Bert
    ingredients_list = list(set(ingredients_dictionary.values()))
    #lenght, common_ings = find_common_ingredients(ingredients_list, bert_vocabulary.splitlines())
    #flavor_cleaned = list({ing for ing in ingredients_list if ing not in common_ings}) ## tot 6491 out of 6608
    # Save a JSON file for training
    with open(flavor_path_json, "w") as json_file:
        json.dump(ingredients_list, json_file, indent=2)

    ## Extend Bert original with Flavor ingredients and save the file
    extend_vocabulary(bert_path, ingredients_list)


    ## Training Model
    args = [
        "--output_dir={}/foodbert/mlm_output".format(path_dir),
        "model_output",
        "--model_type=bert",
        "--model_name=bert-base-cased",
        "--vocabulary_path={}/foodbert/data/extended-bert-base-cased-original.txt".format(path_dir),
        "--ingredients_list={}/foodbert/data/ingredient_names_valid.json".format(path_dir),
        "--do_train",
        "--train_data_file={}/data/train_instructions_clean_valid.txt".format(path_dir),
        "--do_eval",
        "--eval_data_file={}/data/test_instructions_clean_valid.txt".format(path_dir),
        "--mlm",
        "--line_by_line",
        "--per_gpu_train_batch_size=8",
        "--gradient_accumulation_steps=2",
        "--per_gpu_eval_batch_size=8",
        "--save_total_limit=5",
        "--save_steps=10000",
        "--logging_steps=10000",
        "--evaluate_during_training"
    ]
    sys.argv.extend(args)
    main()
