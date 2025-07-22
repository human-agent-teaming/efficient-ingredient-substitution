import json
import os
import pickle
import re
from collections import defaultdict
from typing import List, Dict, Any, Set, Union
from pathlib import Path
import shutil
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')


def find_common_ingredients(list1: List[Any], list2: List[Any]) -> List[List[Any]]:
    set1 = set(list1)
    set2 = set(list2)
    common_ingredients = set1.intersection(set2)
    return len(common_ingredients), common_ingredients


def extend_vocabulary(path_txt: Path, ingredient_list: List):
    dir_path = Path(path_txt).parent
    copy_path = dir_path / f"extended-{Path(path_txt).name}"
    shutil.copy(path_txt, copy_path)

    with open(copy_path, 'r') as f:
        bert_vocabulary = f.read().splitlines()

    try:
        new_words = set(ingredient_list.keys()) - set(bert_vocabulary)

    except:
        new_words = set(ingredient_list) - set(bert_vocabulary)

    if new_words:
        with open(copy_path, 'w') as f:
            f.write('\n'.join(bert_vocabulary + list(new_words)))


def replace_plurals(instruction: str) -> str:
    tokens = nltk.word_tokenize(instruction)
    lemmas = [WordNetLemmatizer().lemmatize(token, pos='n') for token in tokens]
    return ' '.join(lemmas)


class IngredientsCleaner:
    def __init__(self):
        self.regex_patterns = [
            (r'[\[_\\\d%\]]+', ' '),  # replace special characters
            (r'[^a-zA-Z\s]+', ' '),  # replace non-alphanumeric
            (r'[-_]+[-/_]*[-_]*', ' '),  # replace specific patterns
            (r'[\s_]+', ' '),  # replace consecutive whitespace or underscores
            (r'(\b)_([a-zA-Z0-9]+)', r'\2'),  # avoid words starting with particular characters
        ]

    def clean_string(self, text: Union[str, Set[str]], use_underscores: bool, remove_plurals: bool) -> \
            Union[str, Set[str]]:
        if isinstance(text, str):
            cleaned_text = text.strip().lower()
            for pattern, replacement in self.regex_patterns:
                cleaned_text = re.sub(pattern, replacement, cleaned_text)

            words = cleaned_text.split()
            cleaned_words = []
            for word in words:
                cleaned_words.append(replace_plurals(word))

                ##for pattern, replacement in self.plural_patterns:
                #    if remove_plurals and re.search(pattern, word):
            #         word = re.sub(pattern, replacement, word)
            #         break  # Stop after the first match
            # cleaned_words.append(word)

            cleaned_text = ' '.join(cleaned_words)

            # Apply underscores if the flag is set
            if use_underscores:
                cleaned_text = cleaned_text.replace(' ', '_')

            return cleaned_text
        elif isinstance(text, set):
            cleaned_set = {self.clean_string(item, use_underscores, remove_plurals) for item in text}
            return cleaned_set

    def clean_ingredients_names(self, vocabulary: Union[Dict[str, str], List[str], str], use_underscores: bool = False,
                                remove_plurals: bool = True) -> Union[Dict[str, str], List[str]]:
        if isinstance(vocabulary, dict):
            cleaned_dict = {}
            for k, v in vocabulary.items():
                cleaned_key = self.clean_string(k, use_underscores, remove_plurals)
                cleaned_value = self.clean_string(v, use_underscores, remove_plurals)
                cleaned_dict[cleaned_key] = cleaned_value
            return cleaned_dict
        elif isinstance(vocabulary, list) or isinstance(vocabulary, set):
            cleaned_list = [self.clean_string(item, use_underscores, remove_plurals) for item in vocabulary]
            return cleaned_list
        elif isinstance(vocabulary, str):
            cleaned_ingr = self.clean_string(vocabulary, use_underscores, remove_plurals)
            return cleaned_ingr
        else:
            raise TypeError(f"Input must be either a dictionary, list, or string, not {type(vocabulary).__name__}")


def generate_food_dictionary(path: Path):
    with Path(os.path.join(path, 'FoodBert/data/ingredient_names.json')).open() as f:
        food_items = json.load(f)
        food_items_set = set(food_items)

    with Path(os.path.join(path, 'data/train_instructions_clean.txt')).open() as f:
        train_instruction_sentences = f.read().splitlines()
        train_instruction_sentences = [s for s in train_instruction_sentences if
                                       len(s.split()) <= 100]  # remove overlong sentences

    with Path(os.path.join(path, 'data/test_instructions_clean.txt')).open() as f:
        test_instruction_sentences = f.read().splitlines()
        test_instruction_sentences = [s for s in test_instruction_sentences if
                                      len(s.split()) <= 100]  # remove overlong sentences

    instruction_sentences = train_instruction_sentences + test_instruction_sentences
    food_to_sentences_dict = defaultdict(list)
    for sentence in tqdm(instruction_sentences, desc="Processing sentences", unit="sentence"):
        words = re.sub("[^\w]-'", " ", sentence).split()
        for word in words:
            if word in food_items_set:
                food_to_sentences_dict[word].append(sentence)

    with open(os.path.join(path, 'FoodBert/data/food_sentence_dict.pickle'), 'wb') as file:
        pickle.dump(food_to_sentences_dict, file)

