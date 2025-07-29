<h1 align="center"> Stop replacing salt with sugar!
<div>
<img src=https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white>
<img src=https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge&logo=PyTorch>
</h1>


### This repository contains the code and instructions for reproducing the experiments presented in:

> Stop replacing salt with sugar!: Towards Intuitive Human-Agent Teaching", Nikolaos Kondylidis, Andrea Rafanelli, Ilaria Tiddi, Annette Ten Teije, Frank Van Harmelen, 
  Presented at the 22nd European Conference on Multi-Agent Systems (EUMAS 2025), Bucharest.
<p align="center">
    <img src="images/AI-chef.png" height="500" width="500">
</p>

## 1 Experiment Preparation
### 1.1 Download or clone this git
### 1.2 Create python environment and install required libraries

    python3 -m venv int_ingr_sub
    source int_ingr_sub/bin/activate
    pip install -r requirements.txt 

### 1.3 Download the Recipe1MSubs dataset

    mkdir Recipe1MSubs_data
    wget https://dl.fbaipublicfiles.com/gismo/train_comments_subs.pkl
    mv train_comments_subs.pkl Recipe1MSubs_data/train_comments_subs.pkl
    wget https://dl.fbaipublicfiles.com/gismo/val_comments_subs.pkl
    mv val_comments_subs.pkl Recipe1MSubs_data/val_comments_subs.pkl
    wget https://dl.fbaipublicfiles.com/gismo/test_comments_subs.pkl
    mv test_comments_subs.pkl Recipe1MSubs_data/test_comments_subs.pkl

### 2. Run Experiments

### 2.1 Random Tutoring Policy :
#### (Experiments presented in Figure 3)

    # Baseline
    python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=Baseline student.input_representation.ingredient_representation=1_hot_embeddings

    # Accumulative | 1-hot
    python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=Accumulative student.input_representation.ingredient_representation=1_hot_embeddings
    # Accumulative | 1-hot & FoodOn
    python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=Accumulative student.input_representation.ingredient_representation=1_hot_embeddings_w_foodOn
    # Accumulative | Flavorgraph
    python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=Accumulative student.input_representation.ingredient_representation=flavorgraph_embed
    # Accumulative | FoodBert
    python main.py experiment.num_repetitions=4 ingredients.vocabulary=foodbert teacher.active_learning_method=Random student.learning_method=Accumulative student.input_representation.ingredient_representation=foodbert_embed

    # P_Networks | 1-hot
    python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=P_Networks student.input_representation.ingredient_representation=1_hot_embeddings
    # P_Networks | 1-hot & FoodOn
    python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=P_Networks student.input_representation.ingredient_representation=1_hot_embeddings_w_foodOn
    # P_Networks | Flavorgraph
    python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=P_Networks student.input_representation.ingredient_representation=flavorgraph_embed
    # P_Networks | FoodBert
    python main.py experiment.num_repetitions=4 ingredients.vocabulary=foodbert teacher.active_learning_method=Random student.learning_method=P_Networks student.input_representation.ingredient_representation=foodbert_embed

### 2.2 Best performing Learning Methods, for different Tutoring Policies
#### (Experiments presented in Figures 4 and 5)

#### Execution already run before: (with Random Tutoring Policy)
    
    # Random | Baseline
    python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=Baseline student.input_representation.ingredient_representation=1_hot_embeddings
    # Random | Accumulative | 1-hot & FoodOn
    python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Random student.learning_method=Accumulative student.input_representation.ingredient_representation=1_hot_embeddings_w_foodOn
    
#### New executions: (with Balanced Tutoring Policy)

    # Balanced | Baseline
    python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Balanced student.learning_method=Baseline student.input_representation.ingredient_representation=1_hot_embeddings
    # Balanced | Accumulative | 1-hot & FoodOn
    python main.py experiment.num_repetitions=4 ingredients.vocabulary=flavorgraph teacher.active_learning_method=Balanced student.learning_method=Accumulative student.input_representation.ingredient_representation=1_hot_embeddings_w_foodOn




## (Optional) Preprocess Data and (re-)create FoodOn and FoodBERT Embeddings 
### PLEASE NOTE: The scripts and instructions in this section aim to describe the details of our preprocessing, while they are not directly executable.
For example, some datasets and files from other projects need to be downloaded manually in some cases and some paths need to be fixed in the code.
This section of the code will be finalized in case of acceptance.

 
### a. Download FlavorGraph Embeddings

1.  Download FlavorGraph Embeddings ([github](https://github.com/lamypark/FlavorGraph?tab=readme-ov-file), [link](https://drive.google.com/file/d/1MN2dGr-e8x09XSfj0kG4MahTRFY8GDw4/view?usp=sharing))

2.  Move them to "ingredient_vocabularies_and_embeddings/FlavorGraph/"

### a. Download and install the GISMo library
1. Follow the instructions of the [GISMo repository](https://github.com/facebookresearch/gismo/blob/main/gismo/README.md)
2. Download their vocabulary file

        wget https://dl.fbaipublicfiles.com/gismo/vocab_ingrs.pkl
        mv vocab_ingrs.pkl Recipe1MSubs_data/vocab_ingrs.pkl

3. Extract the GISMo Vocabulary and create a new ingredient ID catalogue, together with all ingredient ID <-> name mappings

       python create_Recipe1MSubs_and_FlavorGraph_ingredient_mappings.py

### b. Link to FoodOn and Create FoodOn Embeddings
1. Download FoodOn Ontology files from [here](https://github.com/boschresearch/EaT-PIM/tree/main/data/foodon_ontologies). The "foodon_complete.owl" file is in this git, and it is an older version of [this file](https://raw.githubusercontent.com/FoodOntology/foodon/master/foodon.owl) from the [FoodOn project](https://obofoundry.org/ontology/foodon.html). We don't own the copyrights.
2. Move them to "ingredient_vocabularies_and_embeddings/ontology_properties/foodon_ontologies"
3. Link ingredients to FoodOn. (Code modified from [here](https://github.com/boschresearch/EaT-PIM/blob/main/eatpim/etl/linkers/foodon_matcher.py) and requires Numpy <= 1.21)

       python link_ingredients_to_FoodOn.py

4. Extract FoodOn Properties and Create FoodOn Embeddings File

       python create_FoodOn_embeddings.py


### c. Create FoodBert Embeddings on Recipe1M for FlavorGraph ingredient vocabulary 

1. Download Recipe1M (Link to download is currently broken, please contact the authors ([Recipe1M project page](https://im2recipe.csail.mit.edu/)))
2. Data Preprocessing:
    - **parsing.py**: Creates structured recipe data from raw input.
      ```bash
      python embedding_experiments.parsing.py
      
    - **clean_ingredient_names.py**: Cleans and standardizes ingredient names.
      ```bash
      python embedding_experiments.clean_ingredient_names.py
      
    - **normalise_recipes.py**: Creates a cleaned version of Recipe1M dataset with synonyms from GISMO/FlavorGraph.
      ```bash
      python embedding_experiments.normalise_recipes.py

3. Create GISMO to FlavorGraph mappings:
   ```bash
   python ingredient_vocabularies_and_embeddings/FlavorGraph/create_gismo_to_flavorgraph_ingredient_name_mapping.py
   
5. FoodBERT embeddings run(in foodbert/ directory)
   - **preprocess_instructions.py**: Extracts and prepares instructions from cleaned recipes for training.
     ```bash
     python embedding_experiments.foodbert.preprocess_instructions.py
     
   - **training.py**: Extends BERT vocabulary with food ingredients and trains the model on recipe instructions.
     ```bash
     python embedding_experiments.foodbert.generate_embeddings.py
     
   - **translate_embeddings.py**: Processes and translates the generated embeddings with GISMO vocabulary.
     ```bash
     python embedding_experiments.foodbert.translate_embeddings.py
     
    ##### Note: Some code is adapted from: https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution
   
            
