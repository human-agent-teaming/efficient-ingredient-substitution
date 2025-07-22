
from experiment_utils import *
import os
import pickle
import numpy as np
import numpy.typing as npt

from omegaconf import DictConfig
import random

from typing import Optional

class Student:

    def __init__(self, learning_method:str, training_substitution_inputs: list[tuple[str, set[int], int]],
                 ingredient_vocabulary_size: int, cfg: DictConfig):

        self.learning_method:str = learning_method
        self.training_substitution_inputs: list[tuple[str, set[int], int]] = training_substitution_inputs
        self.ingredient_vocabulary_size: int = ingredient_vocabulary_size
        self.ingredient_embeddings: Optional[npt.NDArray] = None
        self.vector_distance_metric = cfg.student.vector_distance_metric

        self.ingredient_indices: list[int] = list(range(ingredient_vocabulary_size))

        self.source_ingredient_weight: float = cfg.student.input_representation.source_ingredient_weight

        self.recipe_ingredient_aggregation = cfg.student.input_representation.recipe_ingredient_aggregation
        if self.recipe_ingredient_aggregation != 'mean' and self.recipe_ingredient_aggregation != 'sum' and self.recipe_ingredient_aggregation != 'ignore':
            raise ValueError(
                f"Unknown cfg.student.recipe_ingredient_aggregation provided: {self.recipe_ingredient_aggregation}")

        self.set_recipe_ingredient_weights_according_to_params(cfg)

        # create a numpy array where each value is equal to the index. Will be used as a way to access specific rows of tables at the same time
        self.target_ingredient_indices: npt.NDArray = np.asarray(list(range(self.ingredient_vocabulary_size)))

        # we create a dictionary to cache input representations of recipes and source ingredients, not having to calculate them multiple times
        self.cached_input_representation: dict[tuple[frozenset[int], int], np.ndarray] = {}

        # load the corresponding embeddings and order them according to ingredient index
        if cfg.student.input_representation.ingredient_representation == 'flavorgraph_embed':

            with (open(os.path.join(cfg.ingredients.directory, cfg.ingredients.flavorgraph_embeddings_sub_path),
                       "rb")) as openfile:
                flavorgraph_node_embeddings = pickle.load(openfile)

            with (open(os.path.join(cfg.ingredients.directory, cfg.ingredients.new_ingredient_id_to_flavorgraph_id),
                       "rb")) as openfile:
                new_ingredient_id_to_flavorgraph_id = pickle.load(openfile)

            self.ingredient_embeddings: npt.NDArray(float) = np.zeros((len(new_ingredient_id_to_flavorgraph_id), 300))

            for ingredient_index in new_ingredient_id_to_flavorgraph_id:
                flavorgraph_ingredient_index: str = str(new_ingredient_id_to_flavorgraph_id[ingredient_index])
                flavorgraph_embedding = flavorgraph_node_embeddings[flavorgraph_ingredient_index]
                self.ingredient_embeddings[ingredient_index] = flavorgraph_embedding

        elif cfg.student.input_representation.ingredient_representation == 'foodbert_embed':

            with (open(os.path.join(cfg.ingredients.directory, cfg.ingredients.foodbert_embeddings_filename), "rb")) as openfile:
                self.ingredient_embeddings = pickle.load(openfile)

        elif cfg.student.input_representation.ingredient_representation == 'random_embed':
            self.ingredient_embeddings = np.random.uniform(size=(self.ingredient_vocabulary_size, 300))

        elif cfg.student.input_representation.ingredient_representation == '1_hot_embeddings':
            self.ingredient_embeddings = np.diag(np.ones(self.ingredient_vocabulary_size))


        elif cfg.student.input_representation.ingredient_representation == '1_hot_embeddings_w_foodOn':

            food_on_hops:int = cfg.student.input_representation.food_on_hops

            food_on_embeddings_filename:str = cfg.ingredients.new_gismo_idx2food_on_embeddings_path + "_" + str(food_on_hops) + "_hops_quadr_values_WO_blank_nodes.pickle"

            with open(os.path.join(cfg.ingredients.directory, food_on_embeddings_filename), 'rb') as handle:
                new_gismo_to_food_on_classes_embeddings = pickle.load(handle)

                assert new_gismo_to_food_on_classes_embeddings.shape[0] == self.ingredient_vocabulary_size

            ingredient_1_hot_embeddings = np.diag(np.ones(self.ingredient_vocabulary_size))

            self.ingredient_embeddings = np.concatenate((ingredient_1_hot_embeddings, new_gismo_to_food_on_classes_embeddings), axis = 1)

        else:
            raise ValueError("Unknown ingredient representation provided to the student: " + cfg.student.input_representation.ingredient_representation)

        if self.learning_method == 'Baseline':
            self.substitution_frequency: npt.NDArray[np.int32] = np.zeros((self.ingredient_vocabulary_size, self.ingredient_vocabulary_size), dtype=int)

        elif self.learning_method == 'P_Networks':
            # keeping track of how many times each ingredient appears as target ingredient, to see which have prototypical clusters and to calculate averages
            self.target_ingredients_counter: npt.NDArray[int] = np.zeros(self.ingredient_vocabulary_size, dtype=int)
            self.target_ingredients_prototypical_cluster_centroids: npt.NDArray = np.zeros((self.ingredient_vocabulary_size, self.ingredient_embeddings.shape[-1]), dtype=float)

        elif self.learning_method == 'Accumulative':
            representation_dimensionality:int = self.ingredient_embeddings.shape[-1]

            self.target_ingredient_scores: npt.NDArray = np.zeros(
                (self.ingredient_vocabulary_size, representation_dimensionality), dtype=float)
        else:
            raise ValueError(f"Unknown learning method provided: {self.learning_method}")

    def set_recipe_ingredient_weights_according_to_params(self, cfg:DictConfig) -> None:

        self.recipe_ingredient_weights: npt.NDArray
        # if all recipe ingredients should have the same weight, when used to represent a recipe
        if cfg.student.input_representation.recipe_ingredient_weights == 'uniform':
            self.recipe_ingredient_weights = np.ones(self.ingredient_vocabulary_size, dtype=int)
        elif cfg.student.input_representation.recipe_ingredient_weights == 'popularity':
            # if the recipe ingredient weight's should reflect the ingredient popularity in the dataset:
            # (n/R), n: times this ingredient appears in a recipe, R: total number of recipes
            ingredient_frequencies, number_of_recipes = self.get_ingredient_frequencies_in_recipes_and_number_of_recipes()
            self.recipe_ingredient_weights = ingredient_frequencies
            # self.recipe_ingredient_weights = ingredient_frequencies / number_of_recipes
        elif cfg.student.input_representation.recipe_ingredient_weights == 'exp_popularity':
            ingredient_frequencies, number_of_recipes = self.get_ingredient_frequencies_in_recipes_and_number_of_recipes()
            popularity_values: npt.NDArray = ingredient_frequencies / number_of_recipes
            exp_popularity_values: npt.NDArray = np.power(10, popularity_values)
            self.recipe_ingredient_weights = exp_popularity_values
            # # normalize values by dividing with maximum value
            # max_popularity = np.max(exp_popularity_values)
            # self.recipe_ingredient_weights = np.divide(exp_popularity_values, max_popularity,
            #                                            out=np.zeros_like(ingredient_frequencies),
            #                                            where=ingredient_frequencies != 0)
        elif cfg.student.input_representation.recipe_ingredient_weights == 'rarity':
            # if the recipe ingredient weight's should reflect the ingredient rarity in the dataset:
            # (1/n), n: times this ingredient appears in a recipe
            # Equal to 0 if n is 0
            ingredient_frequencies, _ = self.get_ingredient_frequencies_in_recipes_and_number_of_recipes()
            self.recipe_ingredient_weights = np.divide(1, ingredient_frequencies,
                                                       out=np.zeros_like(ingredient_frequencies),
                                                       where=ingredient_frequencies != 0)
        elif cfg.student.input_representation.recipe_ingredient_weights == 'log_rarity':
            ingredient_frequencies, number_of_recipes = self.get_ingredient_frequencies_in_recipes_and_number_of_recipes()
            log_rarity_values: npt.NDArray = np.divide(number_of_recipes, ingredient_frequencies,
                                                       out=np.zeros_like(ingredient_frequencies),
                                                       where=ingredient_frequencies != 0)
            self.recipe_ingredient_weights = log_rarity_values
            # # normalize values by dividing with maximum value
            # max_rarity = np.max(log_rarity_values)
            # self.recipe_ingredient_weights = np.divide(log_rarity_values, max_rarity,
            #                                            out=np.zeros_like(ingredient_frequencies),
            #                                            where=ingredient_frequencies != 0)
        else:
            raise ValueError(
                "cfg.student.input_representation.recipe_ingredient_weights unknown: " + cfg.student.input_target.recipe_ingredient_weights)

    def get_ingredient_frequencies_in_recipes_and_number_of_recipes(self) -> tuple[npt.NDArray[np.float32], int]:

        recipe_ingredients: dict[str, set[int]] = dict()


        for recipe_id, ingredients_in_recipe, _ in self.training_substitution_inputs:
            if recipe_id not in recipe_ingredients:
                recipe_ingredients[recipe_id] = ingredients_in_recipe

        number_of_recipes: int = len(recipe_ingredients)

        ingredient_frequencies: npt.NDArray[np.float32] = np.zeros(self.ingredient_vocabulary_size, dtype=np.float32)

        for recipe_id in recipe_ingredients:
            for ingredient in recipe_ingredients[recipe_id]:
                ingredient_frequencies[ingredient] += 1

        return ingredient_frequencies, number_of_recipes

    def get_input_representation(self, recipe_ingredients: set[int], source_ingredient: int) -> npt.NDArray:


        # check if input representation is cached
        hashable_input:tuple[frozenset[int], int] = (frozenset(recipe_ingredients), source_ingredient)
        if hashable_input in self.cached_input_representation:
            return self.cached_input_representation[hashable_input]

        input_representation: npt.NDArray
        source_ingredient_representation: npt.NDArray = self.ingredient_embeddings[source_ingredient]

        # in case we ignore the recipe ingredients (besides the source ingredient), then we only return the source ingredient representation
        if self.recipe_ingredient_aggregation == 'ignore':
            input_representation = source_ingredient_representation
        else:

            if self.source_ingredient_weight == 'as_recipe_ingr':
                ingredients: list[int] = list(recipe_ingredients)

                ingredient_representations: npt.NDArray = self.ingredient_embeddings[ingredients]

                ingredient_weights: npt.NDArray = self.recipe_ingredient_weights[ingredients]

                all_ingredient_normalized_weights: npt.NDArray = ingredient_weights / np.sum(ingredient_weights)

                all_ingredient_normalized_weights_reshaped: npt.NDArray = all_ingredient_normalized_weights.reshape(-1,
                                                                                                                    1)
                weighted_representation: npt.NDArray = ingredient_representations * all_ingredient_normalized_weights_reshaped

                input_representation = np.sum(weighted_representation, axis=0)

            else:

                remaining_recipe_ingredients: list[int] = list(recipe_ingredients - {source_ingredient})
                # retrieve embeddings of remaining recipe ingredients
                remaining_recipe_ingredient_representations: npt.NDArray = self.ingredient_embeddings[remaining_recipe_ingredients]

                all_ingredient_representations: npt.NDArray = np.vstack([remaining_recipe_ingredient_representations, source_ingredient_representation])

                remaining_recipe_ingredient_weights: npt.NDArray = self.recipe_ingredient_weights[remaining_recipe_ingredients]

                # the weight of the source ingredient is equal to the total weight of the other recipe ingredients, times the multiplier provided as a parameter
                source_ingredient_weight: float = self.source_ingredient_weight * np.sum(remaining_recipe_ingredient_weights)

                all_ingredient_weights: npt.NDArray = np.append(remaining_recipe_ingredient_weights, np.array([source_ingredient_weight]), axis=0)

                all_ingredient_normalized_weights: npt.NDArray = all_ingredient_weights / np.sum(all_ingredient_weights)

                all_ingredient_normalized_weights_reshaped: npt.NDArray = all_ingredient_normalized_weights.reshape(-1, 1)

                weighted_representation: npt.NDArray = all_ingredient_representations * all_ingredient_normalized_weights_reshaped

                input_representation = np.sum(weighted_representation, axis=0)

        # save calculated input representation to cache dictionary
        self.cached_input_representation[hashable_input] = input_representation

        return input_representation


    def get_remaining_recipe_representation(self, recipe_ingredients: set[int], source_ingredient: int, aggregation:str) -> npt.NDArray:
        remaining_recipe_ingredients: list[int] = list(recipe_ingredients - {source_ingredient})

        # in case the recipe does not have any other ingredients, we use the source ingredient to represent the whole recipe
        if len(remaining_recipe_ingredients) == 0:
            raise ValueError("Recipe ingredients is only the source ingredient !")
        # retrieve embeddings of recipe ingredients
        remaining_recipe_ingredient_representations: npt.NDArray = self.ingredient_embeddings[remaining_recipe_ingredients]

        if aggregation == 'mean':
            recipe_representation: npt.NDArray = np.mean(remaining_recipe_ingredient_representations, axis=0)
        elif aggregation == 'sum':
            recipe_representation: npt.NDArray = np.sum(remaining_recipe_ingredient_representations, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method provided: {aggregation}")

        return recipe_representation

    # the student learning policy
    def learn_from_substitution_example(self, substitution_input: tuple[str, frozenset[int], int], target_ingredient:int) -> None:

        recipe_id: str = substitution_input[0]
        recipe_ingredients_frozenset: frozenset[int] = substitution_input[1]
        recipe_ingredients: set[int] = set(recipe_ingredients_frozenset)
        source_ingredient: int = substitution_input[2]

        if self.learning_method == 'Baseline':
            self.substitution_frequency[source_ingredient, target_ingredient] += 1

        elif self.learning_method == 'P_Networks':
            # we retrieve the embedding of the source ingredient
            # source_ingredient_embedding = self.ingredient_embeddings[source_ingredient]
            input_representation = self.get_input_representation(recipe_ingredients, source_ingredient)

            # we increase the counter of this ingredient used as a target ingredient
            self.target_ingredients_counter[target_ingredient] += 1
            # we update the cluster centroid of this target ingredient embedding to reflect the new average embedding of its source ingredients
            self.target_ingredients_prototypical_cluster_centroids[target_ingredient] += \
                (input_representation - self.target_ingredients_prototypical_cluster_centroids[target_ingredient]) / self.target_ingredients_counter[target_ingredient]

        elif self.learning_method == 'Accumulative':

            # we calculate the input representation (recipe + source, ingredient according to experiment parameters)
            input_representation = self.get_input_representation(recipe_ingredients=recipe_ingredients, source_ingredient=source_ingredient)

            # we add this input representation to the "scores" of the target ingredient, allowing us to "relate" them in the future
            self.target_ingredient_scores[target_ingredient] += input_representation

        return None

    def suggest_substitution(self, substitution_input: tuple[str, set[int], int]) -> list[int]:
        recipe_id: str = substitution_input[0]
        recipe_ingredients: set[int] = substitution_input[1]
        source_ingredient: int = substitution_input[2]

        if self.learning_method == 'Baseline':
            ranked_substitution_candidates = np.flip(np.argsort(self.substitution_frequency[source_ingredient]))

            complete_set_of_ingredients_ordered = self.add_remaining_ingredients_at_the_end_of_list_in_random_order(
                ordered_relevant_target_ingredients_indices=list(ranked_substitution_candidates))
            return complete_set_of_ingredients_ordered

        elif self.learning_method == 'P_Networks':
            # see which ingredients have been used as target ingredients before
            evidenced_target_ingredients_mask: npt.NDArray = self.target_ingredients_counter > 0
            # retrieve their indices
            candidate_target_ingredients: npt.NDArray = self.target_ingredient_indices[evidenced_target_ingredients_mask]
            # retrieve their embeddings (the embedding of their cluster centroid in the prototypical network, not the embedding of the target ingredient)
            candidate_target_ingredients_prototypical_cluster_centroids: npt.NDArray = self.target_ingredients_prototypical_cluster_centroids[evidenced_target_ingredients_mask]

            input_representation = self.get_input_representation(recipe_ingredients, source_ingredient)

            # order candidate target ingredients and return their indices
            ordered_vector_indices: npt.NDArray
            ordered_vector_indices, _ = self.get_ordered_vector_indices_from_closer_to_further(
                vector_to_minimize_distance_to=input_representation,
                vector_candidates=candidate_target_ingredients_prototypical_cluster_centroids)

            # we get the original ingredient indices according to their ordering in the subset of candidates
            ordered_relevant_target_ingredients: list[int] = list(candidate_target_ingredients[ordered_vector_indices])

            complete_set_of_ingredients_ordered = self.add_remaining_ingredients_at_the_end_of_list_in_random_order(
                ordered_relevant_target_ingredients_indices=ordered_relevant_target_ingredients)

            return complete_set_of_ingredients_ordered

        elif self.learning_method == 'Accumulative':
            # we calculate the input representation (recipe + source, ingredient according to experiment parameters)
            input_representation: npt.NDArray = self.get_input_representation(recipe_ingredients=recipe_ingredients,
                                                                 source_ingredient=source_ingredient)

            # create a mask of candidate target ingredients that have any overlap with current input representation, in the representation space
            candidate_targets_mask = np.dot(self.target_ingredient_scores, input_representation) != 0

            ordered_target_ingredient_indices_list: list[int]

            # if there are no candidates:
            if np.sum(candidate_targets_mask) == 0:
                ordered_target_ingredient_indices_list = []
            else:
                # retrieve the indices of these candidate target ingredients
                candidate_ingredient_indices: npt.NDArray = self.target_ingredient_indices[candidate_targets_mask]
                # retrieve their representation again
                candidate_vector_representations: npt.NDArray = self.target_ingredient_scores[candidate_targets_mask]

                # calculate inner product to represent similarity scores between input representation and candidate target ingredient representations
                inner_products: npt.NDArray = np.dot(candidate_vector_representations, input_representation.T)

                # order their vector representations according to similarity with current input representation
                ordered_candidate_index_positions: npt.NDArray = np.argsort(inner_products)

                ordered_candidate_indices: npt.NDArray = candidate_ingredient_indices[ordered_candidate_index_positions]

                # we flip the order of candidate rankings (so that maximum similarity goes first, since we calculate similarity score and not minimizing distances)
                ordered_target_ingredient_indices_list = list(np.flip(ordered_candidate_indices))

            # put the remaining target ingredient indices at the end of the resulted ordered candidate indices list
            complete_candidate_ingredient_results_ordered: list[int] = self.add_remaining_ingredients_at_the_end_of_list_in_random_order(
                ordered_relevant_target_ingredients_indices=list(ordered_target_ingredient_indices_list))

            # return complete set of target ingredient indices as results
            return complete_candidate_ingredient_results_ordered

        else:
            raise ValueError(f"Unknown learning method provided: {self.learning_method}")

    def add_remaining_ingredients_at_the_end_of_list_in_random_order(self, ordered_relevant_target_ingredients_indices:list[int]) -> list[int]:

        # in order to rank all ingredients of the vocabulary (and fairly be compared using ranking metrics) we retrieve the remaining ones
        irrelevant_target_ingredients = list(
            set(range(self.ingredient_vocabulary_size)) - set(ordered_relevant_target_ingredients_indices))

        # shuffle remaining target ingredients from the ingredient vocabulary
        random.shuffle(irrelevant_target_ingredients)
        # print(f"Results: {len(ordered_relevant_target_ingredients) / self.ingredient_vocabulary_size:7.5}" )

        # then put them at the end of the results list
        complete_set_of_ingredients_ordered: list[
            int] = ordered_relevant_target_ingredients_indices + irrelevant_target_ingredients

        return complete_set_of_ingredients_ordered

    def get_ordered_vector_indices_from_closer_to_further(self, vector_to_minimize_distance_to: npt.NDArray,
                                                          vector_candidates: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:

        assert vector_candidates.shape[0] > 0

        distances: npt.NDArray
        if self.vector_distance_metric == "cosine_distance":

            # Compute the magnitude of each row (axis=1)
            magnitudes = np.linalg.norm(vector_candidates, axis=1, ord=None, keepdims=True)
            # normalize the embeddings of the multiple ingredients
            multiple_ingredient_embeddings_normalized = vector_candidates / magnitudes

            # Compute the magnitude of each row (axis=1)
            magnitude = np.linalg.norm(vector_to_minimize_distance_to, ord=None, keepdims=True)
            # normalize the embeddings of the single ingredient
            unit_vector = vector_to_minimize_distance_to / magnitude

            # calculate their cosine similarity
            cosine_similarities = np.dot(unit_vector, multiple_ingredient_embeddings_normalized.T)
            # translate cosine similarities to normalized distances
            distances = 1 - cosine_similarities

        else:

            distances: npt.NDArray = vector_candidates - vector_to_minimize_distance_to

            if self.vector_distance_metric == 'l1':
                distances = np.abs(distances)
            elif self.vector_distance_metric == 'l2':
                distances = distances ** 2
            elif self.vector_distance_metric == 'l4':
                distances = distances ** 4
            elif self.vector_distance_metric == 'sqrt':
                distances = np.sqrt(np.abs(distances))
            else:
                raise ValueError("Parameter student.vector_distance_metric not identified!: " + self.vector_distance_metric)

            distances = np.sum(distances, axis=1)


        return np.argsort(distances), np.sort(distances)




def save_student_to_file(student:Student, used_examples:list[tuple[tuple[str, frozenset[int], int], int]], save_dir:str) -> None:
    import os
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/saved_student_after_examples_{len(used_examples)}.pickle", 'wb') as handle:
        pickle.dump(student, handle)
    with open(f"{save_dir}/list_of_examples_used_at_{len(used_examples)}.pickle", 'wb') as handle:
        pickle.dump(used_examples, handle)

def load_student_from_file(examples_used:int, save_dir:str) -> tuple[Student, list[tuple[tuple[str, frozenset[int], int], int]]]:
    with (open(f"{save_dir}/saved_student_after_examples_{examples_used}.pickle", "rb")) as openfile:
        student:Student = pickle.load(openfile)
    with (open(f"{save_dir}/list_of_examples_used_at_{examples_used}.pickle", "rb")) as openfile:
        used_examples:list[tuple[tuple[str, frozenset[int], int], int]] = pickle.load(openfile)
        return student, used_examples
