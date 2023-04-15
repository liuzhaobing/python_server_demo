# -*- coding:utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModel

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class Model:
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    para_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    para_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    print(f"load model [{MODEL_NAME}]")

    @classmethod
    def calculate_sentence_embeddings(cls, sentences):
        encoded_input = cls.para_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = cls.para_model(**encoded_input)
        # Perform pooling. In this case, max pooling.
        sentence_embeddings = cls.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @classmethod
    def calculate_cosine(cls, sentence_embedding1, sentence_embedding2):
        return torch.nn.functional.cosine_similarity(sentence_embedding1, sentence_embedding2, dim=0).item()


if __name__ == '__main__':
    sentences = ["介绍一下中国移动", "中国移动的介绍"]
    embeddings = Model.calculate_sentence_embeddings(sentences)
    print(Model.calculate_cosine(embeddings[0], embeddings[1]))
