# -*- coding:utf-8 -*-
import logging
from typing import Union, List

import numpy
import torch
from sentence_transformers import SentenceTransformer

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class Model:
    MODEL_NAME = "GanymedeNil/text2vec-large-chinese"
    model = SentenceTransformer(MODEL_NAME,
                                device="cuda" if torch.cuda.is_available() else "cpu",
                                cache_folder=".huggingface_cache")
    logging.info(f"load model [{MODEL_NAME}]")

    def embedding(self, sentence: Union[str, List[str]]):
        """
        Get sentence embedding
        :param sentence: sentence
        :return: sentence embedding
        """
        return self.model.encode(sentence)

    def calculate_cosine(self, sentence_embedding1, sentence_embedding2):
        """ Calculate cosine similarity between two vectors
        :param sentence_embedding1: Tensor 1
        :param sentence_embedding2: Tensor 2
        :return: cosine similarity
        """
        if isinstance(sentence_embedding1, numpy.ndarray):
            sentence_embedding1 = torch.from_numpy(sentence_embedding1)
        if isinstance(sentence_embedding2, numpy.ndarray):
            sentence_embedding2 = torch.from_numpy(sentence_embedding2)
        return torch.nn.functional.cosine_similarity(sentence_embedding1, sentence_embedding2, dim=0).item()


model = Model()

if __name__ == '__main__':
    sentences = ["介绍一下中国移动", "中国移动的介绍"]
    embeddings = model.embedding(sentences)
    print(model.calculate_cosine(embeddings[0], embeddings[1]))
