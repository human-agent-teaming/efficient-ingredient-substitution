import hydra
from omegaconf import DictConfig
import os

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # set maximum number of threads used by some libraries including numpy according to the value of the provided parameter
    os.environ["OMP_NUM_THREADS"] = str(cfg.experiment.num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cfg.experiment.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(cfg.experiment.num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cfg.experiment.num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cfg.experiment.num_threads)
    # then import the experiment.py file that imports numpy
    from experiment import Experiment

    experiment = Experiment(recipe1Msubs_path=cfg.dataset.recipe1Msubs_path, cfg=cfg)

    print("Teacher params:\n", cfg.teacher)
    print("Student params:\n", cfg.student)

    experiment.run_repeatable_experiments(num_repetitions=cfg.experiment.num_repetitions, eval_in_steps=cfg.experiment.eval_in_steps,
                               num_threads=cfg.experiment.num_threads, eval_split=cfg.experiment.eval_on, cfg=cfg)


if __name__ == "__main__":
    main()

### Random Tutoring Policy :
#### (Experiments presented in Figure 2)

# python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=Baseline student.input_representation.ingredient_representation=1_hot_embeddings

# Accumulative | 1-hot
# python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=Accumulative student.input_representation.ingredient_representation=1_hot_embeddings
# Accumulative | 1-hot & FoodOn
# python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=Accumulative student.input_representation.ingredient_representation=1_hot_embeddings_w_foodOn
# Accumulative | Flavorgraph
# python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=Accumulative student.input_representation.ingredient_representation=flavorgraph_embed
# Accumulative | FoodBert
# python main.py experiment.num_repetitions=4 ingredients.vocabulary=foodbert teacher.active_learning_method=Random student.learning_method=Accumulative student.input_representation.ingredient_representation=foodbert_embed

# Accumulative | 1-hot
# python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=P_Networks student.input_representation.ingredient_representation=1_hot_embeddings
# Accumulative | 1-hot & FoodOn
# python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=P_Networks student.input_representation.ingredient_representation=1_hot_embeddings_w_foodOn
# Accumulative | Flavorgraph
# python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=P_Networks student.input_representation.ingredient_representation=flavorgraph_embed
# Accumulative | FoodBert
# python main.py experiment.num_repetitions=4 ingredients.vocabulary=foodbert teacher.active_learning_method=Random student.learning_method=P_Networks student.input_representation.ingredient_representation=foodbert_embed

### Best performing Learning Methods, for different Tutoring Policies
#### (Experiments presented in Figures 3 and 4)

# Common executions with before: (with random Tutoring Policy)

# Random | Baseline
# python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=Baseline student.input_representation.ingredient_representation=1_hot_embeddings
# Accumulative | 1-hot & FoodOn
# python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=Accumulative student.input_representation.ingredient_representation=1_hot_embeddings_w_foodOn

# New executions: (With Balanced Tutoring Policy)
# Balanced | Baseline
# python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Balanced student.learning_method=Baseline student.input_representation.ingredient_representation=1_hot_embeddings
# Balanced | Accumulative | 1-hot & FoodOn
# python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Balanced student.learning_method=Accumulative student.input_representation.ingredient_representation=1_hot_embeddings_w_foodOn
