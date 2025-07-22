import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
from tqdm import tqdm
from embedding_experiments.utils.vocabulary import IngredientsCleaner, replace_plurals


class ParsingRecipes:
    def __init__(self, recipe_list: List[Dict[str, Any]], ingredient_list: List[Dict[str, Any]], cleaning:bool,
                 use_underscores:bool, remove_plurals:bool):
        self.recipe_list = recipe_list
        self.ingredient_list = ingredient_list
        self.cleaning = cleaning
        self.use_underscores = use_underscores
        self.remove_plurals = remove_plurals

    def replace_ingredients(self, text: str, valid_ingredients: List[str]) -> Tuple[Union[str, Any], List[str]]:

        # partial matches word by word
        words = re.findall(r'\b\w+\b', text)
        replaced_words = set()  # to avoid replacing the same word twice

        for i, word in enumerate(words):
            if word.lower() in replaced_words:
                continue

            best_match = self.find_best_match(word, valid_ingredients)
            if best_match and best_match[1] > 0:
                if self.use_underscores:
                    # replace the matched word with the underscored ingredient
                    text = re.sub(rf'\b{re.escape(word)}\b', best_match[0].replace(' ', '_'), text, flags=re.IGNORECASE)
                    replaced_words.add(word.lower())
                    valid_ingredients.remove(best_match[0].replace(' ', '_'))  # remove the ingredient finded
                else:
                    # replace the matched word with the non-underscored ingredient
                    text = re.sub(rf'\b{re.escape(word)}\b', best_match[0], text, flags=re.IGNORECASE)
                    replaced_words.add(word.lower())
                    valid_ingredients.remove(best_match[0])  # remove the ingredient finded

        return text, valid_ingredients

    def get_valid_ingredients(self, recipe_id: str) -> List[str]:

        for recipe in self.ingredient_list:
            if recipe['id'] == recipe_id:
                return [
                    ingredient['text']
                    for ingredient, valid in zip(recipe['ingredients'], recipe['valid'])
                    if valid
                ]
        return []

    def update_recipe_instructions(self, recipe_id: str) -> Optional[Dict[str, Any]]:

        for recipe in tqdm(self.recipe_list, desc="Processing Recipes", unit="recipe"):
            if recipe['id'] == recipe_id:
                valid_ingredients = self.get_valid_ingredients(recipe_id)
                if self.cleaning:
                    valid_ingredients = IngredientsCleaner().clean_ingredients_names(valid_ingredients,
                                                                                     use_underscores=self.use_underscores,
                                                                                     remove_plurals=self.remove_plurals)
                updated_instructions = []
                for instruction in recipe['instructions']:
                    instruction['text'] = replace_plurals(instruction['text'])  #avoid plurals
                    updated_text, valid_ingredients = self.replace_ingredients(instruction['text'], valid_ingredients)
                    updated_text = self.remove_repeating_phrases(updated_text)
                    updated_instructions.append({'text': updated_text})

                updated_recipe = recipe.copy()  # create a copy to avoid modifying the original
                updated_recipe['instructions'] = updated_instructions
                return updated_recipe
        return None

    def process_all_recipes(self) -> List[Dict[str, Any]]:
        updated_recipes = []
        for recipe in self.recipe_list:
            updated_recipe = self.update_recipe_instructions(recipe['id'])
            if updated_recipe:
                updated_recipes.append(updated_recipe)
        return updated_recipes

    @staticmethod
    def find_best_match(word: str, valid_ingredients: List[str]) -> Optional[Tuple[str, int]]:
        """
        find the valid ingredient that best matches a given word
        """
        word_set = set(word.lower().split())
        best_match = None
        max_common_words = 0
        for ingredient in valid_ingredients:
            if '_' in ingredient:
                ingredient = ingredient.replace('_', ' ')
            ingredient_set = set(ingredient.lower().split())
            common_words = len(word_set & ingredient_set)
            if common_words > max_common_words or (common_words == max_common_words and len(ingredient_set) < len(
                    best_match[0].split()) if best_match else float('inf')):
                max_common_words = common_words
                best_match = (ingredient, common_words)
        return best_match

    @staticmethod
    def remove_repeating_phrases(text: str, max_words: int = 3) -> str:
        words = text.split()
        result = []
        i = 0
        while i < len(words):
            phrase = words[i]
            next_index = i + 1

            # to find the longest repeating phrase
            for j in range(min(max_words, len(words) - i), 0, -1):
                candidate = ' '.join(words[i:i + j])
                if ' '.join(words[i + j:]).startswith(candidate + ' '):
                    phrase = candidate
                    next_index = i + 2 * j
                    break

            result.append(phrase)
            i = next_index

        return ' '.join(result)

    def export_updated_recipes(self, output_path: str) -> None:

        updated_recipes = self.process_all_recipes()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(updated_recipes, f, ensure_ascii=False, indent=4)

        print(f"Updated recipes exported to: {output_path}")


if __name__ == '__main__':
    recipe1m_json_path: Path = Path('embedding_experiments/data/layer1.json')
    recipe1m_valid_path: Path = Path('embedding_experiments/data/det_ingrs.json')
    export_path: Path = Path('embedding_experiments/data/recipe1m_instruction_valid_ingredients.json')

    with recipe1m_json_path.open() as file:
        recipe1m = json.load(file)

    with recipe1m_valid_path.open() as file:
        recipe1m_valid = json.load(file)

    parser = ParsingRecipes(recipe1m, recipe1m_valid, cleaning=True, use_underscores=True, remove_plurals=True)
    parser.export_updated_recipes(export_path)
