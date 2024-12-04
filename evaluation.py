import pandas as pd
import numpy as np
from typing import List
from nltk.stem import WordNetLemmatizer
import utils
import os
import json
import utils

class Evaluator():
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.metrics_name = None

    def lemmatize(self, word:str):  #小文字化も含む
        return self.wnl.lemmatize(word.lower())
    
    def eval(self, words:List[str]|str):
        pass

    def eval_topics(self, topics:List[List[str]]):
        scores = []
        for top_words in topics:
            score_of_a_topic = self.eval(top_words)
            scores.append(score_of_a_topic)
        return utils.mean(scores), scores
    
class CoverageEvaluator(Evaluator):
    def __init__(self, reference_documents_file_path):
        super().__init__()
        self.metrics_name = f"Cov_in_{reference_documents_file_path}"
        self.reference_documents = utils.read_texts(reference_documents_file_path)
    
    def eval_topics(self, topics: List[List[str]]):
        num_documets_including_top_words = 0
        for document in self.reference_documents:
            correspond_topics = set()
            for exist_word in document.split(" "):
                for iof_topic, top_words in enumerate(topics):
                    if exist_word in top_words: correspond_topics.add(iof_topic)
            if(len(correspond_topics)) > 0: num_documets_including_top_words += 1
        return num_documets_including_top_words / len(self.reference_documents), None
    
class FactualityEvaluator(Evaluator):
    def __init__(self, reference_vocab_file_path):
        super().__init__()
        self.metrics_name = f"Fct_in_{reference_vocab_file_path}"
        self.reference_vocab = utils.read_texts(reference_vocab_file_path)
    
    def eval_topics(self, topics: List[List[str]]):
        concatinated_top_words_set = set([word for top_words in topics for word in top_words])
        num_words_present_in_vocab = 0
        for word in concatinated_top_words_set:
            if word in self.reference_vocab: num_words_present_in_vocab += 1
        return num_words_present_in_vocab / len(concatinated_top_words_set), None


class ConcreatnessEvaluator(Evaluator):
    def __init__(self, conc_dict_file_path="Concreteness_ratings_Brysbaert_et_al_BRM.txt"):
        super().__init__()
        self.conc_df = pd.read_csv(conc_dict_file_path, sep="\t", header=0)
    
    def can_eval(self, word):
        return True if word in self.conc_df["Word"].values else False
    
    def eval(self, words:List[str]|str, statistics_function):
        if isinstance(words, str): words = [words]
        conc_values = []
        for word in words:
            lemmatized_word = self.lemmatize(word)
            if self.can_eval(lemmatized_word):
                conc_value = self.conc_df[self.conc_df["Word"]==lemmatized_word]["Conc.M"].values[0]
                conc_values.append(conc_value)
            else:
                conc_values.append(None)
        return statistics_function(conc_values)

class MeanConcreatnessEvaluator(ConcreatnessEvaluator):
    def __init__(self, conc_dict_file_path="Concreteness_ratings_Brysbaert_et_al_BRM.txt"):
        super().__init__(conc_dict_file_path)
        self.metrics_name = "MeanConc"
    
    def eval(self, words:List[str]|str):
        super().eval(words, utils.mean)


class MaxConcreatnessEvaluator(ConcreatnessEvaluator):
    def __init__(self, conc_dict_file_path="Concreteness_ratings_Brysbaert_et_al_BRM.txt"):
        super().__init__(conc_dict_file_path)
        self.metrics_name = "MaxConc"
    
    def eval(self, words:List[str]|str):
        super().eval(words, utils.max)


class MinConcreatnessEvaluator(ConcreatnessEvaluator):
    def __init__(self, conc_dict_file_path="Concreteness_ratings_Brysbaert_et_al_BRM.txt"):
        super().__init__(conc_dict_file_path)
        self.metrics_name = "MinConc"
    
    def eval(self, words:List[str]|str):
        super().eval(words, utils.min)


class EvaluationSystem():
    def __init__(self, evaluators):
        self.evaluators = evaluators
        self.scores = dict()

    def read_topics(self, outputs_dir_path) -> List[List[str]]:
        top_words_lines = utils.read_texts(os.path.join(outputs_dir_path, "top_words.txt"))
        return [top_words_line.split(" ") for top_words_line in top_words_lines]
    
    def eval(self, words:List[str]|str):
        for evaluator in self.evaluators:
            score, _ = evaluator.eval_topics(words)
            self.scores[evaluator.metrics_name] = score

    def save(self, outputs_dir_path):
        with open(os.path.join(outputs_dir_path, "scores.json"), "w") as f:
            json.dump(self.scores, f)