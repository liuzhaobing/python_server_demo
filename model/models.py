# -*- coding:utf-8 -*-
import sentence_transformers
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class QQSimNew:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.para_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.para_model = AutoModel.from_pretrained(self.model_name).to(device)
        print(f"load model [{self.model_name}]")

    def calculate_sentence_embeddings(self, sentences):
        encoded_input = self.para_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.para_model(**encoded_input)
        # Perform pooling. In this case, max pooling.
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    def calculate_cosine(sentence_embeddings):
        import torch.nn.functional as F
        query_embedding = sentence_embeddings[0]
        score_list = []
        for qa_embedding in sentence_embeddings[1:]:
            score = F.cosine_similarity(query_embedding, qa_embedding, dim=0).item()
            score_list.append(score)
        return score_list

    def calculate_similarity(self, sentences):
        sentence_embeddings = self.calculate_sentence_embeddings(sentences)
        return self.calculate_cosine(sentence_embeddings)

    def call(self, *args, **kwargs):
        sentences = [str(kwargs["sentence1"]), str(kwargs["sentence2"])]
        score = self.calculate_similarity(sentences)
        return score[0]


class HuggingFaceSBertPq:
    def __init__(self):
        self.model_name = 'inkoziev/sbert_pq'
        self.model = sentence_transformers.SentenceTransformer(self.model_name).to(device)

    def call(self, *args, **kwargs):
        sentences = [str(kwargs["sentence1"]), str(kwargs["sentence2"])]
        embeddings = self.model.encode(sentences)
        s = sentence_transformers.util.cos_sim(a=embeddings[0], b=embeddings[1])
        return s.item()


if __name__ == '__main__':
    # qqsim_new = QQSimNew()
    # final_score = qqsim_new.call(sentence1="介绍一下中国移动", sentence2="中国移动的介绍")
    # print(final_score)
    test = HuggingFaceSBertPq()
    final_score = test.call(sentence1="介绍一下中国移动", sentence2="中国移动的介绍")
    print(final_score)
