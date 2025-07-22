
import numpy.typing as npt
import numpy as np
import random
from collections import defaultdict

from typing import Optional

from experiment_utils import *

from omegaconf import DictConfig

class Teacher:

    def __init__(self, active_learning_method:str, substitution_example_inputs: list[tuple[str, set[int], int]],
                 substitution_example_targets: list[int], ingredient_vocabulary_size:int, cfg: DictConfig):
        self.active_learning_method:str = active_learning_method
        self.substitution_example_inputs: list[tuple[str, set[int], int]] = substitution_example_inputs
        self.substitution_example_targets: list[int] = substitution_example_targets
        self.ingredient_vocabulary_size:int = ingredient_vocabulary_size

        # we create a dictionary to cache input representations of recipes and source ingredients, not having to calculate them multiple times
        self.cached_input_representation: dict[tuple[str, frozenset[int], int], np.ndarray] = {}

        self.remaining_examples: list[tuple[tuple[str, set[int], int], int]] = list()
        self.used_examples: set[tuple[tuple[str, frozenset[int], int], int]] = set()

        # Gathering all substitution examples in a list
        for example_index in range(len(self.substitution_example_inputs)):
            substitution_example_input = self.substitution_example_inputs[example_index]
            hashable_substitution_example_input = create_hashable_example_input(substitution_example_input)
            substitution_example_target = self.substitution_example_targets[example_index]

            substitution_example: tuple[tuple[str, frozenset[int], int], int] = (hashable_substitution_example_input, substitution_example_target)

            self.remaining_examples.append(substitution_example)

        # Shuffling the order of the examples
        random.shuffle(self.remaining_examples)

        if self.active_learning_method == "Balanced":

            original_remaining_examples: list[tuple[Union[
                tuple[str, set[int], int], tuple[str, frozenset[int], int]], int]] = self.remaining_examples.copy()

            self.remaining_examples: list[
                tuple[Union[tuple[str, set[int], int], tuple[str, frozenset[int], int]], int]] = []

            substitution_pair_matrix: npt.NDArray = np.zeros(
                (self.ingredient_vocabulary_size, self.ingredient_vocabulary_size), dtype=int)

            pair_ingredient_catalogue: dict[
                tuple[int, int], set[tuple[tuple[str, frozenset[int], int], int]]] = defaultdict(set)

            for substitution_example in original_remaining_examples:
                hashable_substitution_example_input, target_ingredient = substitution_example

                source_ingredient: int = hashable_substitution_example_input[2]

                substitution_pair_matrix[source_ingredient, target_ingredient] += 1

                # source_ingredient_catalogue[source_ingredient][target_ingredient].add(substitution_example)
                pair_ingredient_catalogue[(source_ingredient, target_ingredient)].add(substitution_example)

            examples_per_iteration = dict()

            iteration = 0

            while np.sum(substitution_pair_matrix) > 0:

                ingredient_pairs_with_remaining_examples_mask = substitution_pair_matrix != 0

                iteration_specific_substitution_pair_matrix: npt.NDArray = np.zeros_like(substitution_pair_matrix)

                iteration_specific_substitution_pair_matrix[ingredient_pairs_with_remaining_examples_mask] = np.floor(np.log2(substitution_pair_matrix[ingredient_pairs_with_remaining_examples_mask])) + 1

                examples_to_add_from_this_iteration = list()

                while np.sum(iteration_specific_substitution_pair_matrix) != 0:

                    size_of_largets_buckets = np.max(iteration_specific_substitution_pair_matrix)
                    ingredient_pairs_to_add = iteration_specific_substitution_pair_matrix == size_of_largets_buckets
                    source_ingredients, target_ingredients = np.where(ingredient_pairs_to_add)
                    for source_ingredient, target_ingredient in zip(source_ingredients, target_ingredients):
                        next_example = pair_ingredient_catalogue[(source_ingredient, target_ingredient)].pop()

                        iteration_specific_substitution_pair_matrix[source_ingredient, target_ingredient] -= 1
                        substitution_pair_matrix[source_ingredient, target_ingredient] -= 1

                        self.remaining_examples.append(next_example)

                        examples_to_add_from_this_iteration.append(next_example)

                        original_remaining_examples.remove(next_example)


                examples_per_iteration[iteration] = examples_to_add_from_this_iteration
                iteration += 1

        else:
            if self.active_learning_method != "Random":
                raise ValueError(f"Active learning method provided not recognised: {self.active_learning_method}")


    # the human tutoring policy
    def provide_next_example(self) -> Optional[tuple[tuple[str, frozenset[int], int], int]]:

        if len(self.remaining_examples) == 0:
            return None

        selected_example:tuple[tuple[str, Union[frozenset[int], set[int]], int], int] = self.remaining_examples[0]

        self.remaining_examples.remove(selected_example)

        self.used_examples.add(selected_example)

        return selected_example
