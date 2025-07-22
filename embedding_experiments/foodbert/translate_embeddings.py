import pickle
import numpy.typing as npt
import numpy as np
from typing import Dict, List


class FoodBertEmbeddings:
    def __init__(self):
        with open("embedding_experiments/foodbert/embeddings/foodbert_embeddings_new.pickle", "rb") as openfile:
            self.all_foodbert_embeddings = pickle.load(openfile)
        with open(
                "embedding_experiments/vocabulary/gismo_ingredient_name_to_flavorgraph_ingredient_name_dict_cleaned_new_version.pickle",
                "rb") as f:
            self.synonym_dictionary: Dict[str, str] = pickle.load(f)
        with open(
                "embedding_experiments/vocabulary/gismo_names_mapping.pickle",
                "rb") as f:
            self.mapping_dictionary: Dict[str, str] = pickle.load(f)
        self.merge_embeddings()
        self.create_index_mappings()
        self.save_embeddings_and_mappings()

    def merge_dicts(self, synonym_dictionary: Dict[str, str], embedding_dictionary: Dict[str, List],
                    mapping_dictionary : Dict[str,str]):
        new_embedding_dictionary: Dict[str, List] = {}
        for key, value in synonym_dictionary.items():
            if value in embedding_dictionary:
                original_names = [k for k, v in mapping_dictionary.items() if v == key]
                for original_name in original_names:
                    new_embedding_dictionary[original_name] = embedding_dictionary[value]

        return new_embedding_dictionary

    def merge_embeddings(self):
        self.all_foodbert_embeddings = self.merge_dicts(self.synonym_dictionary, self.all_foodbert_embeddings,
                                                        self.mapping_dictionary)

    def create_index_mappings(self):
        self.foodbert_name_to_index: Dict[str, int] = dict()
        self.foodbert_index_to_name: Dict[int, str] = dict()
        self.embeddings_list: List[npt.NDArray] = list()

        for index, ingredient_name in enumerate(self.all_foodbert_embeddings):
            average_ingredient_embedding = np.mean(self.all_foodbert_embeddings[ingredient_name], axis=0)
            self.foodbert_name_to_index[ingredient_name] = index
            self.foodbert_index_to_name[index] = ingredient_name
            self.embeddings_list.append(average_ingredient_embedding)

        self.foodbert_embeddings: npt.NDArray = np.array(self.embeddings_list)

    def save_embeddings_and_mappings(self):
        with open("embedding_experiments/foodbert/embeddings/files/foodbert_embeddings.pickle", "wb") as openfile:
            pickle.dump(self.foodbert_embeddings, openfile)

        with open("embedding_experiments/foodbert/embeddings/files/foodbert_name_to_index.pickle", "wb") as openfile:
            pickle.dump(self.foodbert_name_to_index, openfile)

        with open("embedding_experiments/foodbert/embeddings/files/foodbert_index_to_name.pickle", "wb") as openfile:
            pickle.dump(self.foodbert_index_to_name, openfile)

        print("Total embeddings in FoodBert:", len(self.foodbert_name_to_index))


if __name__ == "__main__":
    foodbert_embeddings = FoodBertEmbeddings()
