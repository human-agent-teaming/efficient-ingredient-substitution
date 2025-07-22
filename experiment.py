
from experiment_utils import *
from typing import Union, Optional

from student import Student, save_student_to_file
from teacher import Teacher

import numpy as np
import numpy.typing as npt

from omegaconf import DictConfig, OmegaConf
import omegaconf

import datetime

from tqdm.contrib.concurrent import thread_map
import os

import random


import pickle

class Experiment:

    def __init__(self, recipe1Msubs_path: str, cfg: DictConfig):

        self.cfg: DictConfig = cfg

        if cfg.student.input_representation.ingredient_representation == "flavorgraph_embed" and cfg.ingredients.vocabulary != "flavorgraph":
            raise ValueError("Student can't use flavorgraph embeddings if the selected vocabulary is not flavorgraph")
        if cfg.student.input_representation.ingredient_representation == "foodbert_embed" and cfg.ingredients.vocabulary != "foodbert":
            raise ValueError("Student can't use foodbert embeddings if the selected vocabulary is not foodbert")

        if cfg.ingredients.vocabulary == 'flavorgraph':
            ingredient_idx_to_name_dict_path = os.path.join(cfg.ingredients.directory, cfg.ingredients.new_gismo_idx2name_filename)
            ingredient_name_to_idx_dict_path = os.path.join(cfg.ingredients.directory, cfg.ingredients.name2new_gismo_idx_filename)
        elif cfg.ingredients.vocabulary == 'foodbert':
            ingredient_idx_to_name_dict_path = os.path.join(cfg.ingredients.directory, cfg.ingredients.foodbert_index2name_filename)
            ingredient_name_to_idx_dict_path = os.path.join(cfg.ingredients.directory, cfg.ingredients.foodbert_name2index_filename)

        else:
            raise ValueError(f'Unknown ingredient vocabulary: {cfg.ingredients.vocabulary}')

        # load ingredient index to ingredient name mappings
        with (open(ingredient_idx_to_name_dict_path, "rb")) as openfile:
            self.ingredient_idx_to_name_dict: dict[int, Union[str, list[str]]] = pickle.load(openfile)

        with (open(ingredient_name_to_idx_dict_path, "rb")) as openfile:
            self.ingredient_name_to_idx_dict: dict[str, int] = pickle.load(openfile)

        print(f"Vocabulary ingredient names: {len(self.ingredient_name_to_idx_dict)}, vocabulary of ingredient IDs: {len(self.ingredient_idx_to_name_dict)}")

        self.ingredient_vocabulary_size:int = len(self.ingredient_idx_to_name_dict)

        # load the Recipe1MSubs benchmark
        recipe_subs_data = load_recipe_1MSubs_dataset(recipe1Msubs_path)
        # Translate the benchmark so that ingredient synonyms are omitted and all ingredients are used once.
        self.translated_recipe_subs_data: dict[str, dict[str, list[Union[int, tuple[str, set[int], int]]]]] = translate_recipe1MSubs_data_using_gismo_ingredient_IDs(recipe_subs_data, self.ingredient_name_to_idx_dict)


    def run_repeatable_experiments(self, num_repetitions: int, eval_in_steps: Union[int, list[int]], num_threads: int,
                                   eval_split: str, cfg: DictConfig):

        total_training_steps = len(self.translated_recipe_subs_data["train"]["targets"])

        # if it is an integer, we use it as a step
        if isinstance(eval_in_steps, int):
            eval_in_steps = [i for i in range(0, total_training_steps, eval_in_steps)] + [total_training_steps]
        # in case it is a list, then it specifies exactly in which steps to run evaluation
        elif isinstance(eval_in_steps, omegaconf.listconfig.ListConfig):
            if -1 in eval_in_steps:
                eval_in_steps.remove(-1)
                eval_in_steps.append(total_training_steps)

        total_evaluation_metrics_per_evaluation_step: dict[int, dict[str, list[float]]] = {}

        for i in range(num_repetitions):
            print(f"Experiment repetition {i+1} of {num_repetitions} begins:")

            # set random seeds
            random_seed = cfg.experiment.random_seeds[i]
            random.seed(random_seed)
            np.random.seed(random_seed)

            # instantiate Teacher agent
            teacher: Teacher = Teacher(active_learning_method=cfg.teacher.active_learning_method,
                                       substitution_example_inputs=self.translated_recipe_subs_data["train"]["inputs"],
                                       substitution_example_targets=self.translated_recipe_subs_data["train"][
                                           "targets"], ingredient_vocabulary_size=self.ingredient_vocabulary_size,
                                       cfg=cfg)

            # instantiate Student agent
            student: Student = Student(learning_method=cfg.student.learning_method,
                                       training_substitution_inputs=self.translated_recipe_subs_data["train"]["inputs"],
                                       ingredient_vocabulary_size=self.ingredient_vocabulary_size, cfg=cfg)

            # preprocess input for evaluation function
            evaluation_sample_inputs = self.translated_recipe_subs_data[eval_split]["inputs"]
            evaluation_sample_targets = self.translated_recipe_subs_data[eval_split]["targets"]
            student_evaluation_inputs = []
            for j in range(len(evaluation_sample_inputs)):
                student_evaluation_inputs.append((student, evaluation_sample_inputs[j], evaluation_sample_targets[j]))

            # run one experiment
            evaluation_metrics_per_step: dict[int, dict[str, float]] = self.run_experiment(teacher=teacher,
                                                                                           student=student,
                                                                                           eval_in_steps=eval_in_steps,
                                                                                           num_threads=num_threads,
                                                                                           eval_split=eval_split,
                                                                                           student_evaluation_inputs=student_evaluation_inputs,
                                                                                           cfg=cfg)

            # add the student's performance according to the evaluation metrics per examples provided, for this experiment
            total_evaluation_metrics_per_evaluation_step = add_results_dictionary_values_to_overall_results_dictionary(
                total_evaluation_metrics_per_evaluation_step=total_evaluation_metrics_per_evaluation_step,
                evaluation_metrics_per_step=evaluation_metrics_per_step)
            print()

            cfg.experiment.save_student = False

        mean_eval_metrics_pers_step: dict[int, dict[str, float]]
        std_eval_metrics_pers_step: dict[int, dict[str, float]]
        mean_eval_metrics_pers_step, std_eval_metrics_pers_step = aggregate_experiment_repetition_results(total_evaluation_metrics_per_evaluation_step)

        print()
        print(f"Aggregated results on split set '{eval_split}', over {num_repetitions} repetitions, per evaluation step:")

        for eval_in_step in evaluation_metrics_per_step:
            hit_at_1_mean = mean_eval_metrics_pers_step[eval_in_step]["hit_at_1"]
            hit_at_3_mean = mean_eval_metrics_pers_step[eval_in_step]["hit_at_3"]
            hit_at_10_mean = mean_eval_metrics_pers_step[eval_in_step]["hit_at_10"]
            hit_at_20_mean = mean_eval_metrics_pers_step[eval_in_step]["hit_at_20"]
            hit_at_50_mean = mean_eval_metrics_pers_step[eval_in_step]["hit_at_50"]
            hit_at_100_mean = mean_eval_metrics_pers_step[eval_in_step]["hit_at_100"]
            mrr_mean = mean_eval_metrics_pers_step[eval_in_step]["mrr"]

            hit_at_1_std = std_eval_metrics_pers_step[eval_in_step]["hit_at_1"]
            hit_at_3_std = std_eval_metrics_pers_step[eval_in_step]["hit_at_3"]
            hit_at_10_std = std_eval_metrics_pers_step[eval_in_step]["hit_at_10"]
            hit_at_20_std = std_eval_metrics_pers_step[eval_in_step]["hit_at_20"]
            hit_at_50_std = std_eval_metrics_pers_step[eval_in_step]["hit_at_50"]
            hit_at_100_std = std_eval_metrics_pers_step[eval_in_step]["hit_at_100"]
            mrr_std = std_eval_metrics_pers_step[eval_in_step]["mrr"]

            print(
                f'Aggregated repetitions: {num_repetitions}, Provided examples:{eval_in_step:6}, Hit@1: {hit_at_1_mean:6.4} ({hit_at_1_std:6.4}), ' +
                f'Hit@3: {hit_at_3_mean:6.4} ({hit_at_3_std:6.4}), Hit@10: {hit_at_10_mean:6.4} ({hit_at_10_std:6.4}), ' +
                f'Hit@20: {hit_at_20_mean:6.4} ({hit_at_20_std:6.4}), Hit@50: {hit_at_50_mean:6.4} ({hit_at_50_std:6.4}), ' +
                f'Hit@100: {hit_at_100_mean:6.4} ({hit_at_100_std:6.4}), MRR: {mrr_mean:6.4} ({mrr_std:6.4})')

        save_student = False


    def run_experiment(self, teacher:Teacher, student:Student, eval_in_steps: list[int], num_threads, eval_split:str, student_evaluation_inputs, cfg:DictConfig) -> dict[int, dict[str, float]]:

        evaluation_metrics_per_step: dict[int, dict[str, float]] = {}

        maximum_number_of_steps = max(eval_in_steps)

        number_of_used_examples:int = 0

        used_examples:list[tuple[tuple[str, frozenset[int], int], int]] = []

        while number_of_used_examples < maximum_number_of_steps:

            # the teacher can provide its own example
            provided_example: Optional[tuple[tuple[str, frozenset[int], int], int]] = teacher.provide_next_example()

            if provided_example is not None:
                example_input, example_target = provided_example
                # provide the example to the student to learn from it
                student.learn_from_substitution_example(substitution_input=example_input,
                                                             target_ingredient=example_target)
                number_of_used_examples += 1
                used_examples.append(provided_example)

            if cfg.experiment.save_student and len(used_examples) in cfg.experiment.save_student_at_steps:
                save_student_to_file(student, used_examples, cfg.experiment.student_save_dir)

            # check if we need to run get student's evaluation for this step
            if number_of_used_examples in eval_in_steps or (provided_example is None):
                hit_at_1, hit_at_3, hit_at_10, hit_at_20, hit_at_50, hit_at_100, mrr, evaluation_print_statement = self.evaluate_student_performance(
                    split=eval_split, examples_provided=number_of_used_examples,
                    student_evaluation_inputs=student_evaluation_inputs, num_threads=num_threads)

                evaluation_metrics_per_step[number_of_used_examples] = \
                    {"hit_at_1": hit_at_1, "hit_at_3": hit_at_3, "hit_at_10": hit_at_10, "hit_at_20": hit_at_20, "hit_at_50": hit_at_50, "hit_at_100": hit_at_100,
                     "mrr": mrr}

                print(evaluation_print_statement)

            if provided_example is None:
                break

        return evaluation_metrics_per_step

    def evaluate_student_performance(self, split, examples_provided,
                                     student_evaluation_inputs: list[tuple[Student, tuple[str, set[int], int], int]],
                                     num_threads:int):
        # keep track of time it takes to evaluate
        start_time = datetime.datetime.now()

        if num_threads == 0:
            students_rankings_of_correct_suggestions = []
            for student, substitution_input, substitution_target in student_evaluation_inputs:
                students_rankings_of_correct_suggestions.append(student.suggest_substitution(substitution_input).index(substitution_target))
        else:
            # run evaluation in parallel using multiple threads
            students_rankings_of_correct_suggestions = thread_map(evaluate_student_multithread, student_evaluation_inputs,
                                                                  max_workers=num_threads, desc=f"Evaluating Student after {examples_provided} provided examples", leave=False)
        # calculate how much time the evaluation took
        end_time = datetime.datetime.now()
        time_difference = end_time - start_time

        # increase indices of candidate target IDs by 1, so that the first one has rank 1 and not 0
        target_rankings: npt.NDArray[np.int32] = np.asarray(students_rankings_of_correct_suggestions,
                                                            dtype=np.int32) + 1

        number_of_samples = len(student_evaluation_inputs)

        hit_at_1: float = np.sum(target_rankings == 1) * 100 / number_of_samples
        hit_at_3: float = np.sum(target_rankings <= 3) * 100 / number_of_samples
        hit_at_10: float = np.sum(target_rankings <= 10) * 100 / number_of_samples
        hit_at_20: float = np.sum(target_rankings <= 20) * 100 / number_of_samples
        hit_at_50: float = np.sum(target_rankings <= 50) * 100 / number_of_samples
        hit_at_100: float = np.sum(target_rankings <= 100) * 100 / number_of_samples

        mrr: float = np.sum(1 / target_rankings) * 100 / number_of_samples

        print_statement = f'{split} | Provided examples:{examples_provided:6}, Hit@1: {hit_at_1:6.4}, Hit@3: {hit_at_3:6.4}, Hit@10: {hit_at_10:6.4}, Hit@20: {hit_at_20:6.4}, Hit@50: {hit_at_50:6.4}, Hit@100: {hit_at_100:6.4}, MRR: {mrr:6.4} | Time: {time_difference.total_seconds()}'

        return hit_at_1, hit_at_3, hit_at_10, hit_at_20, hit_at_50, hit_at_100, mrr, print_statement

    def get_ingredient_idx(self, ingredient_name: str) -> int:
        # we check if this ingredient name exists in the vocabulary of ingredients (after having removed '<pad>' and '<end>').
        if ingredient_name not in self.ingredient_name_to_idx_dict:
            raise ValueError(f"Ingredient name '{ingredient_name}' was not found!")
        else:
            return self.ingredient_name_to_idx_dict[ingredient_name]


    def get_ingredient_name(self, ingredient_id: int) -> str:
        if ingredient_id not in self.ingredient_idx_to_name_dict:
            raise ValueError(f"Ingredient ID '{ingredient_id}' was not found!")
        else:
            return self.ingredient_idx_to_name_dict[ingredient_id]


