from abc import ABC, abstractmethod


class Recommender(ABC):
    use_wandb = False

    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self, X_train):
        """ Uses training set (Bags instance) for training """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test):
        """ Recommend items """
        raise NotImplementedError
