""" Baselines """
import numpy as np
from numpy.random import rand

from models.base import Recommender
from models.datasets import BagsWithVocab, Bags
from rank_bm25 import BM25Okapi

class RandomBaseline(Recommender):
    """ Random Baseline """

    def __str__(self):
        return "RNDM baseline"

    def train(self, X):
        pass

    def predict(self, X):
        X = X.tocsr()
        random_predictions = rand(X.shape[0], X.shape[1])
        return random_predictions


class Countbased(Recommender):
    """ Item Co-Occurrence """
    def __init__(self, order=1):
        super().__init__()
        self.order = order

    def __str__(self):
        s = "Count-based Predictor"
        s += " (order {})".format(self.order)
        return s

    def train(self, X):
        X = X.tocsr()
        # Construct cooccurrence matrix
        self.cooccurences = X.T @ X
        for __ in range(0, self.order - 1):
            self.cooccurences = self.cooccurences.T @ self.cooccurences

    def predict(self, X):
        # Sum up values of coocurrences
        X = X.tocsr()
        return X @ self.cooccurences


class MostPopular(Recommender):
    """ Most Popular """
    def __init__(self):
        self.most_popular = None

    def __str__(self):
        return "Most Popular baseline"

    def train(self, X):
        X = X.tocsr()
        x_sum = X.sum(0)
        self.most_popular = x_sum

    def predict(self, X):
        return np.broadcast_to(self.most_popular, X.size())


class BM25Baseline(Recommender):
    """ BM25 Baseline """

    def __str__(self):
        return "BM25 Baseline"

    def train(self, X):
        self.corpus = X.get_single_attribute_vocab('title')
        self.tokenized_corpus = [doc.split(" ") for doc in self.corpus.values()]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        pass

    def predict(self, X):
        predictions = list()
        queries = X.owner_attributes['title']
        for query_token in queries:
            tokenized_query = queries[query_token].split(" ")
            doc_scores = self.bm25.get_scores(tokenized_query)
            predictions.append(doc_scores)
        return predictions