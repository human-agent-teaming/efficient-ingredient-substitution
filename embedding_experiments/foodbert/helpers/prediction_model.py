import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

from embedding_experiments.foodbert.helpers.instructions_dataset import InstructionsDataset

"""
This code is taken from the following repository:
https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution
"""


class PredictionModel:

    def __init__(self, model_path: Path, vocab_path: Path, bert_path: Path):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.bert_path = bert_path
        self.model: BertModel = BertModel.from_pretrained(
            pretrained_model_name_or_path=self.model_path)

        with open(self.vocab_path, 'r') as f:
            used_ingredients = json.load(f)
        self.tokenizer = BertTokenizer(vocab_file=self.bert_path, do_lower_case=False,
                                       max_len=128, never_split=used_ingredients)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model.to(self.device)

    def predict_embeddings(self, sentences):
        dataset = InstructionsDataset(tokenizer=self.tokenizer, sentences=sentences)
        dataloader = DataLoader(dataset, batch_size=100, pin_memory=True)

        embeddings = []
        ingredient_ids = []
        for batch in dataloader:
            batch = batch.to(self.device)
            with torch.no_grad():
                embeddings_batch = self.model(batch)
                embeddings.extend(embeddings_batch[0])
                ingredient_ids.extend(batch)

        return torch.stack(embeddings), ingredient_ids

    def compute_embedding_for_ingredient(self, sentence, ingredient_name):
        embeddings, ingredient_ids = self.predict_embeddings([sentence])
        embeddings_flat = embeddings.view((-1, 768))
        ingredient_ids_flat = torch.stack(ingredient_ids).flatten()
        food_id = self.tokenizer.convert_tokens_to_ids(ingredient_name)
        food_embedding = embeddings_flat[ingredient_ids_flat == food_id].cpu().numpy()

        return food_embedding[0]
