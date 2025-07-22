import json
import pickle
import re
from pathlib import Path
from embedding_experiments.utils.recipe_normalizer import RecipeNormalizer
from tqdm import tqdm
from typing import List, Dict, Set, Union, Tuple

"""
This code is adapted from the following repository:
https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution
"""


def match_ingredients(normalized_instruction_tokens: List[str],
                      synonym_dict: Dict[str, Union[str, None]],
                      ingredient_names_set: Set[tuple],
                      n: int) -> List[str]:
    not_word_tokens = ['.', ',', '!', '?', ' ', ';', ':']
    new_instruction_tokens = []
    i = 0
    while i < len(normalized_instruction_tokens):
        sublist = normalized_instruction_tokens[i:i + n]
        if sublist[0] in not_word_tokens or sublist[-1] in not_word_tokens:
            new_instruction_tokens.extend(sublist)
            i += n
            continue
        clean_sublist = tuple([token for token in sublist if token not in not_word_tokens])
        j = 0
        while j < len(clean_sublist):
            item = clean_sublist[j]
            if item in ingredient_names_set:
                new_ingredient = synonym_dict.get(item)
                if new_ingredient is not None:
                    new_instruction_tokens.append(new_ingredient)
                    if j < len(clean_sublist) - 1 and clean_sublist[j + 1] not in not_word_tokens:
                        new_instruction_tokens.append(' ')
                    j += 1
                else:
                    new_instruction_tokens.append(item)
                    if j < len(clean_sublist) - 1 and clean_sublist[j + 1] not in not_word_tokens:
                        new_instruction_tokens.append(' ')
                    j += 1
            else:
                new_instruction_tokens.append(item)
                if j < len(clean_sublist) - 1 and clean_sublist[j + 1] not in not_word_tokens:
                    new_instruction_tokens.append(' ')
                j += 1
        i += n

    return new_instruction_tokens


def normalize_instruction(instruction_doc: List[str],
                          synonym_dict: Dict[str, Union[str, None]],
                          ingredient_names_set: Set[Tuple[str, ...]],
                          instruction_normalizer: RecipeNormalizer) -> str:
    instruction_norm = ''
    for idx, word in enumerate(instruction_doc):
        if not word.is_punct:  # we want a space before all non-punctuation words
            space = ' '
        else:
            space = ''
        if word.tag_ in ['NN', 'NNS', 'NNP', 'NOUN', 'NNPS']:
            instruction_norm += space + instruction_normalizer.lemmatize_token_to_str(token=word,
                                                                                      token_tag='NOUN')
        else:
            instruction_norm += space + word.text

    instruction_norm = instruction_norm.strip()

    normalized_instruction_tokens = re.findall(r"[\w'-]+|[.,!?; ]", instruction_norm)
    for n in range(8, 1, -1):  # stop at 2 because matching tokens with length 1 can stay as they are
        any_match = False
        while True:
            prev_tokens = normalized_instruction_tokens.copy()
            normalized_instruction_tokens = match_ingredients(normalized_instruction_tokens,
                                                              synonym_dict,
                                                              ingredient_names_set,
                                                              n)
            if normalized_instruction_tokens == prev_tokens:
                break
            any_match = True

        if not any_match:
            break
    return ''.join(normalized_instruction_tokens)


if __name__ == '__main__':

    recipe1m_json_path: Path = Path('data/recipe1m_instruction_valid_ingredients.json')
    export_path: Path = Path('data/cleaned_recipe1m_valid_ingrs_normalised.json')

    with open("vocabulary/gismo_ingredient_name_to_flavorgraph_ingredient_name_dict_cleaned_new_version.pickle",
              "rb") as file:
        synonyms_dictionary = pickle.load(file)

    with recipe1m_json_path.open() as f:
        recipes = json.load(f)

    ingredients_set = {(synonym) for synonym in synonyms_dictionary}
    instruction_lists = [recipe['instructions'] for recipe in recipes]
    instructions = []
    for instruction_list in instruction_lists:
        for instruction in instruction_list:
            instructions.append(instruction['text'])
    instruction_normalizer = RecipeNormalizer()
    normalized_instructions = instruction_normalizer.model.pipe(instructions,
                                                                n_process=-1,
                                                                batch_size=1000)

    for recipe in tqdm(recipes, total=len(recipes)):
        for instruction_dict in recipe['instructions']:
            normalized_instruction = normalize_instruction(next(normalized_instructions),
                                                           synonyms_dictionary,
                                                           ingredients_set,
                                                           instruction_normalizer=instruction_normalizer)
            instruction_dict['text'] = normalized_instruction

    with export_path.open('w') as f:
        json.dump(recipes, f)
