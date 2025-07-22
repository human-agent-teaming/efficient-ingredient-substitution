import json
import os
import pickle5 as pickle
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Tuple
from tqdm import tqdm
from embedding_experiments.foodbert.helpers.prediction_model import PredictionModel
from embedding_experiments.utils.vocabulary import generate_food_dictionary

"""
This code is adapted from the following repository:
https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution
"""


def map_ingredients_to_input_ids(model, path: Path) -> Dict[str, int]:
    with Path(os.path.join(path, 'foodbert/data/ingredient_names_valid.json')).open() as f:
        ingredients = json.load(f)

    ingredient_ids = model.tokenizer.convert_tokens_to_ids(ingredients)
    ingredient_ids_dict = dict(zip(ingredients, ingredient_ids))

    return ingredient_ids_dict


def _random_sample_with_min_count(population: List[str], k: int) -> List[str]:
    if len(population) <= k:
        return population
    else:
        return random.sample(population, k)


def sample_random_sentence_dict(food_to_sentences_dict: Dict[str, List[str]], max_sentence_count: int) -> Dict[str, List[str]]:
    # only keep 100 randomly selected sentences
    food_to_sentences_dict_random_samples = {food: _random_sample_with_min_count(sentences, max_sentence_count) for
                                             food, sentences in food_to_sentences_dict.items()}

    return food_to_sentences_dict_random_samples


def embedding_prediction(food_dictionary: Dict[str, List[str]], model, path: Path) -> None:
    all_ingredient_ids = map_ingredients_to_input_ids(model, path)
    food_to_embeddings_dict = defaultdict(list)

    for food, sentences in tqdm(food_dictionary.items(),
                                total=len(food_dictionary),
                                desc='Calculating Embeddings for Food items'):
        embeddings, ingredient_ids = model.predict_embeddings(sentences)
        embeddings_flat = embeddings.view((-1, 768))
        ingredient_ids_flat = torch.stack(ingredient_ids).flatten()
        food_id = all_ingredient_ids[food]
        food_embeddings = embeddings_flat[ingredient_ids_flat == food_id].cpu().numpy()
        food_to_embeddings_dict[food].extend(food_embeddings)

    food_dict = dict()
    for k, v in food_to_embeddings_dict.items():
        try:
            food_dict[k] = np.stack(v)
        except:
            pass

    with open(os.path.join(path, 'foodbert/embeddings/foodbert_embeddings.pickle'), 'wb') as file:
        pickle.dump(food_dict, file)


if __name__ == '__main__':

    path_dir = 'embedding_experiments/'

    if os.path.exists(os.path.join(path_dir, 'foodbert/data/food_sentence_dict.pickle')):
        with open(os.path.join(path_dir, 'foodbert/data/food_sentence_dict.pickle'), 'rb') as f:
            food_to_sentences_dict = pickle.load(f)
    else:
        generate_food_dictionary(path_dir)
        with open(os.path.join(path_dir, 'foodbert/data/food_sentence_dict.pickle'), 'rb') as f:
            food_to_sentences_dict = pickle.load(f)

    food_to_sentences_dict_samples = sample_random_sentence_dict(food_to_sentences_dict, max_sentence_count=100)

    prediction_model = PredictionModel(model_path=os.path.join(path_dir, 'foodbert/mlm_output/checkpoint-700000'),
                                       vocab_path=os.path.join(path_dir, 'foodbert/data/ingredient_names.json'),
                                       bert_path=os.path.join(path_dir, 'foodbert/data/extended-bert-base-cased-original.txt'))

    embedding_prediction(food_dictionary=food_to_sentences_dict_samples,
                         model=prediction_model,
                         path=path_dir)
