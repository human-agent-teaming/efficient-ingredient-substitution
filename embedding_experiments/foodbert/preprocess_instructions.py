import json
import os
from pathlib import Path
from typing import List, Dict, Union

"""
This code is adapted from the following repository:
https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution
"""


def extract_instructions_from_recipes(recipes: List[Dict[str, List[Dict[str, str]]]]) -> List[str]:
    instructions = []

    for recipe in recipes:
        for instruction in recipe['instructions']:
            instructions.append(instruction['text'])

    return instructions


def create_instruction(path: Path):
    recipes_path = os.path.join(path, 'cleaned_recipe1m_valid_ingrs_normalised.json')
    train_instructions_path = os.path.join(path, 'train_instructions_clean_valid.txt')
    test_instructions_path = os.path.join(path, 'test_instructions_clean_valid.txt')

    with open(recipes_path, 'r') as f:
        recipes = json.load(f)

    train_recipes = [item for item in recipes if item["partition"] == "train"]
    test_recipes = [item for item in recipes if item["partition"] in ("test", "val")]
    train_instructions = extract_instructions_from_recipes(train_recipes)
    test_instructions = extract_instructions_from_recipes(test_recipes)

    print(f'Train Instructions: {len(train_instructions)}\n'
          f'Test Instructions: {len(test_instructions)}')

    with open(train_instructions_path, 'w') as f:
        f.write('\n'.join(train_instructions))

    with open(test_instructions_path, 'w') as f:
        f.write('\n'.join(test_instructions))


if __name__ == '__main__':
    path = Path('data/')
    create_instruction(path)
