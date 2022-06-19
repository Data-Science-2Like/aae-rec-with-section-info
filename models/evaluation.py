import pickle
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta
import os
import random
import torch
import sys
from abc import ABC, abstractmethod
from sklearn.preprocessing import minmax_scale
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from models import rank_metrics as rm
from models.datasets import corrupt_sets
from models.transforms import lists2sparse

from utils.log import log


def argtopk(X, k):
    """
    Picks the top k elements of (sparse) matrix X

    >>> X = np.arange(10).reshape(1, -1)
    >>> i = argtopk(X, 3)
    >>> i
    (array([[0]]), array([[9, 8, 7]]))
    >>> X[argtopk(X, 3)]
    array([[9, 8, 7]])
    >>> X = np.arange(20).reshape(2,10)
    >>> ix, iy = argtopk(X, 3)
    >>> ix
    array([[0],
           [1]])
    >>> iy
    array([[9, 8, 7],
           [9, 8, 7]])
    >>> X[ix, iy]
    array([[ 9,  8,  7],
           [19, 18, 17]])
    >>> X = np.arange(6).reshape(2,3)
    >>> X[argtopk(X, 123123)]
    array([[2, 1, 0],
           [5, 4, 3]])
    """
    assert len(X.shape) == 2, "X should be two-dimensional array-like"
    rows = np.arange(X.shape[0])[:, np.newaxis]
    if k is None or k >= X.size:
        ind = np.argsort(X, axis=1)[:, ::-1]
        return rows, ind

    assert k > 0, "k should be positive integer or None"

    ind = np.argpartition(X, -k, axis=1)[:, -k:]
    # sort indices depending on their X values
    cols = ind[rows, np.argsort(X[rows, ind], axis=1)][:, ::-1]
    return rows, cols


class Metric(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, y_true, y_pred, average=True):
        pass


class RankingMetric(Metric):
    """ Base class for all ranking metrics
    may also be used on its own to quickly get ranking scores from Y_true,
    Y_pred pair
    """

    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop('k', None)
        super().__init__()

    def __call__(self, y_true, y_pred, average=True):
        """ Gets relevance scores,
        Sort based on y_pred, then lookup in y_true
        >>> Y_true = np.array([[1,0,0],[0,0,1]])
        >>> Y_pred = np.array([[0.2,0.3,0.1],[0.2,0.5,0.7]])
        >>> RankingMetric(k=2)(Y_true, Y_pred)
        array([[0, 1],
               [1, 0]])
        """
        ind = argtopk(y_pred, self.k)
        rs = y_true[ind]
        return rs


class MRR(RankingMetric):
    """ Mean reciprocal rank at k

    >>> mrr_at_5 = MRR(5)
    >>> callable(mrr_at_5)
    True
    >>> Y_true = np.array([[1,0,0],[0,0,1]])
    >>> Y_pred = np.array([[0.2,0.3,0.1],[0.2,0.5,0.7]])
    >>> MRR(2)(Y_true, Y_pred)
    (0.75, 0.25)
    >>> Y_true = np.array([[1,0,1],[1,0,1]])
    >>> Y_pred = np.array([[0.4,0.3,0.2],[0.4,0.3,0.2]])
    >>> MRR(3)(Y_true, Y_pred)
    (1.0, 0.0)
    """

    def __init__(self, k=None):
        super().__init__(k=k)

    def __call__(self, y_true, y_pred, average=True):
        # compute mrr wrt k
        rs = super().__call__(y_true, y_pred)
        return rm.mean_reciprocal_rank(rs, average=average)


class MAP(RankingMetric):
    """ Mean average precision at k """

    def __init__(self, k=None):
        super().__init__(k=k)

    def __call__(self, y_true, y_pred, average=True):
        """
        >>> Y_true = np.array([[1,0,0],[0,0,1]])
        >>> Y_pred = np.array([[0.2,0.3,0.1],[0.2,0.5,0.7]])
        >>> MAP(2)(Y_true, Y_pred)
        (0.75, 0.25)
        >>> Y_true = np.array([[1,0,1],[1,0,1]])
        >>> Y_pred = np.array([[0.3,0.2,0.3],[0.6,0.5,0.7]])
        >>> MAP(3)(Y_true, Y_pred)
        (1.0, 0.0)
        >>> Y_true = np.array([[1,0,1],[1,1,1]])
        >>> Y_pred = np.array([[0.4,0.3,0.2],[0.4,0.3,0.2]])
        >>> MAP(3)(Y_true, Y_pred)
        (0.9166666666666666, 0.08333333333333337)
        """
        rs = super().__call__(y_true, y_pred)
        if average:
            return rm.mean_average_precision(rs)
        else:
            return np.array([rm.average_precision(r) for r in rs])


class P(RankingMetric):
    def __init__(self, k=None):
        super().__init__(k=k)

    def __call__(self, y_true, y_pred, average=True):
        """
        >>> Y_true = np.array([[1,0,1,0],[1,0,1,0]])
        >>> Y_pred = np.array([[0.2,0.3,0.1,0.05],[0.2,0.5,0.7,0.05]])
        >>> P(2)(Y_true, Y_pred)
        (0.5, 0.0)
        >>> P(4)(Y_true, Y_pred)
        (0.5, 0.0)
        """
        # compute p wrt k
        rs = super().__call__(y_true, y_pred)
        ps = (rs > 0).mean(axis=1)
        if average:
            return ps.mean(), ps.std()
        else:
            return ps


class R(RankingMetric):
    def __init__(self, k=None):
        super().__init__(k=k)

    def __call__(self, y_true, y_pred, average=True):
        rs = super().__call__(y_true, y_pred)
        # we want to know how many true items have been recalled of all true items
        retrieved = (rs > 0).sum(axis=1)
        # print(f"Shape {retrieved.shape}, max: {np.max(retrieved)}, min: {np.min(retrieved)}")

        gold = (y_true > 0).sum(axis=1)

        # maybe there are some sets which don't have any possible keys
        # remove them from the metrics calculation
        retrieved = retrieved[gold > 0]
        gold = gold[gold > 0]
        # gold = np.maximum((y_true > 0).sum(axis=1),np.ones(retrieved.shape[0]))
        # print(f"Shape gold {gold.shape}, max: {np.max(gold)}, min: {np.min(retrieved)}")
        re = np.divide(retrieved, gold)
        if average:
            return re.mean(), re.std()
        else:
            return re


BOUNDED_METRICS = {
    # (bounded) ranking metrics
    '{}@{}'.format(M.__name__.lower(), k): M(k)
    for M in [MRR, MAP, P] for k in [5, 10, 20]
}
BOUNDED_METRICS['P@1'] = P(1)

RECALL_METRICS = {
    '{}@{}'.format(M.__name__.lower(), k): M(k)

    for M in [R] for k in [5, 10, 20, 500, 2000]
}

UNBOUNDED_METRICS = {
    # unbounded metrics
    M.__name__.lower(): M()
    for M in [MRR, MAP]
}

# METRICS = {**BOUNDED_METRICS, **UNBOUNDED_METRICS}
METRICS = {**UNBOUNDED_METRICS, **RECALL_METRICS}


def remove_non_missing(Y_pred, X_test, copy=True):
    """
    Scales the predicted values between 0 and 1 and  sets the known values to
    zero.
    >>> Y_pred = np.array([[0.6,0.5,-1], [40,-20,10]])
    >>> X_test = np.array([[1, 0, 1], [0, 1, 0]])
    >>> remove_non_missing(Y_pred, X_test)
    array([[0.    , 0.9375, 0.    ],
           [1.    , 0.    , 0.5   ]])
    """
    Y_pred_scaled = minmax_scale(Y_pred,
                                 feature_range=(0, 1),
                                 axis=1,  # Super important!
                                 copy=copy)
    # we remove the ones that were already present in the orig set
    Y_pred_scaled[X_test.nonzero()] = 0.
    return Y_pred_scaled


def evaluate(ground_truth, predictions, metrics, batch_size=None):
    """
    Main evaluation function, used by Evaluation class but can also be
    reused to recompute metrics
    """

    n_samples = ground_truth.shape[0]
    assert predictions.shape[0] == n_samples

    metrics = [m if callable(m) else METRICS[m] for m in metrics]

    if batch_size is not None:
        batch_size = int(batch_size)

        # Important: Results consist of Mean + Std dev
        # Add all results per sample to array
        # Average later
        results_per_metric = [[] for _ in range(len(metrics))]
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            pred_batch = predictions[start:end, :]
            gold_batch = ground_truth[start:end, :]
            if sp.issparse(pred_batch):
                pred_batch = pred_batch.toarray()
            if sp.issparse(gold_batch):
                gold_batch = gold_batch.toarray()

            for i, metric in enumerate(metrics):
                results_per_metric[i].extend(metric(gold_batch, pred_batch, average=False))

        results = [(x.mean(), x.std()) for x in map(np.array, results_per_metric)]
    else:
        if sp.issparse(ground_truth):
            ground_truth = ground_truth.toarray()
        if sp.issparse(predictions):
            predictions = predictions.toarray()
        results = [metric(ground_truth, predictions) for metric in metrics]

    return results


def reevaluate(gold_file, predictions_file, metrics):
    """ Recompute metrics from files """
    Y_test = sp.load_npz(gold_file)
    Y_pred = np.load(predictions_file)
    return evaluate(Y_test, Y_pred, metrics)


class Evaluation(object):
    def __init__(self,
                 dataset,
                 year,
                 metrics=METRICS,
                 logfile=sys.stdout,
                 logdir=None,
                 val_year=-1,
                 eval_each=False):
        self.dataset = dataset
        self.year = year
        self.val_year = val_year
        self.metrics = metrics
        self.logfile = logfile
        self.logdir = logdir
        self.eval_each = eval_each  # specifies if evaluation metric should be calculated each epoch

        self.save_model = Path('./serialize/aae.pickle')

        self.train_set, self.test_set = None, None
        self.x_test, self.y_test = None, None
        if self.val_year > 0:
            self.val_set = None
            # self.x_val, self.y_val = None, None

    def setup(self, seed=42, min_elements=1, max_features=None,
              min_count=None, drop=1):

        self.min_elements = min_elements
        self.max_features = max_features
        self.min_count = min_count
        self.drop = drop

        # we could specify split criterion and drop choice here
        """ Splits and corrupts the data accordion to criterion """
        random.seed(seed)
        np.random.seed(seed)

        train_set, test_set, val_set = None, None, None
        if self.val_year > 0:
            train_set, val_set, test_set = self.dataset.train_val_test_split(val_year=self.val_year,
                                                                             test_year=self.year)
        else:
            train_set, test_set = self.dataset.train_test_split(on_year=self.year)

        log("=" * 80)
        log("Train:", train_set)
        if val_set is not None:
            log("Val:", val_set)
        log("Test:", test_set)
        log("Next Pruning:\n\tmin_count: {}\n\tmax_features: {}\n\tmin_elements: {}"
            .format(min_count, max_features, min_elements))
        train_set = train_set.build_vocab(min_count=min_count,
                                          max_features=max_features,
                                          apply=True)
        test_set = test_set.apply_vocab(train_set.vocab)
        if val_set is not None:
            val_set = val_set.apply_vocab(train_set.vocab)

        # Train and test sets are now BagsWithVocab
        train_set.prune_(min_elements=min_elements)
        test_set.prune_(min_elements=min_elements)
        if val_set is not None:
            val_set.prune_(min_elements=min_elements)
        log("Train:", train_set)
        if val_set is not None:
            log("Val:", val_set)
        log("Test:", test_set)
        log("Drop parameter:", drop)

        noisy, missing = corrupt_sets(test_set.data, drop=drop)

        assert len(noisy) == len(missing) == len(test_set)

        test_set.data = noisy
        log("-" * 80)

        # if val_set is not None:
        #    noisy_val, missing_val = corrupt_sets(val_set.data, drop=drop)
        #    assert len(noisy_val) == len(missing_val) == len(val_set)
        #    val_set.data = noisy_val
        #    self.y_val = lists2sparse(missing_val, val_set.size(1)).tocsr(copy=False)
        #    self.x_val = lists2sparse(noisy_val, val_set.size(1)).tocsr(copy=False)
        #
        #    self.val_set = val_set
        self.val_set = val_set

        # THE GOLD
        self.y_test = lists2sparse(missing, test_set.size(1)).tocsr(copy=False)

        self.train_set = train_set
        self.test_set = test_set

        # just store for not recomputing the stuff
        self.x_test = lists2sparse(noisy, train_set.size(1)).tocsr(copy=False)
        return self

    def __call__(self, recommenders, batch_size=None):
        if None in (self.train_set, self.test_set, self.x_test, self.y_test):
            raise UserWarning("Call .setup() before running the experiment")
        if self.val_year > 0 and self.val_set is None:
            raise UserWarning("No validation data found")

        if self.logdir:
            os.makedirs(self.logdir, exist_ok=True)
            vocab_path = os.path.join(self.logdir, "vocab.txt")
            with open(vocab_path, 'w') as vocab_fh:
                print(*self.train_set.index2token, sep='\n', file=vocab_fh)
            gold_path = os.path.join(self.logdir, "gold")
            sp.save_npz(gold_path, self.y_test)

        for recommender in recommenders:

            log(recommender)
            train_set = self.train_set.clone()
            test_set = self.test_set.clone()
            t_0 = timer()
            if self.val_set is not None:
                recommender.train(train_set, self.val_set)
            elif self.eval_each:
                log("Training with callback")
                recommender.train(train_set, eval_each=True, eval_cb=(lambda m : self.metrics_calculation(recommender, m)))
            else:
                recommender.train(train_set)
            log("Training took {} seconds."
                .format(timedelta(seconds=timer() - t_0)))
            # torch.save(recommender.state_dict(), "1epoch_test.model")
            split_metrics_calculation = False
            #if self.save_model:
            #    pickle.dump(recommender, open(self.save_model, "wb"))
            #    log("Serialized to {}".format(self.save_model))
            t_1 = timer()
            total_result = None
            if split_metrics_calculation:
                for start in tqdm(range(0, len(test_set), batch_size)):
                    end = start + batch_size
                    batch = test_set[start:end]
                    y_pred = recommender.predict(batch)
                    if sp.issparse(y_pred):
                        y_pred = y_pred.toarray()
                    else:
                        # dont hide that we are assuming an ndarray to be returned
                        y_pred = np.asarray(y_pred)

                    y_pred = remove_non_missing(y_pred, self.x_test[start:end], copy=True)

                    results = evaluate(self.y_test[start:end], y_pred, metrics=self.metrics, batch_size=batch_size)

                    if total_result == None:
                        total_result = len(batch), results
                    else:
                        old_len, old_results = total_result
                        new_res = []
                        for (old_mean, old_std), (new_mean, new_std) in zip(old_results, results):
                            mean = old_mean * old_len + new_mean * len(batch)
                            mean = mean / (old_len + len(batch))
                            std = old_std
                            new_res.append((mean, std))
                        total_result = old_len + len(batch), new_res

                for metric, (mean, std) in zip(self.metrics, total_result[1]):
                    log("- {}: {} ({})".format(metric, mean, std))

                log("\nOverall time: {} seconds."
                    .format(timedelta(seconds=timer() - t_0)))
                log('-' * 79)


            else:
                y_pred = recommender.predict(test_set)
                if sp.issparse(y_pred):
                    y_pred = y_pred.toarray()
                else:
                    # dont hide that we are assuming an ndarray to be returned
                    y_pred = np.asarray(y_pred)

                # set likelihood of documents that are already cited to zero, so
                # they don't influence evaluation
                y_pred = remove_non_missing(y_pred, self.x_test, copy=True)

                log("Prediction took {} seconds."
                    .format(timedelta(seconds=timer() - t_1)))

                if self.logdir:
                    t_1 = timer()
                    pred_file = os.path.join(self.logdir, repr(recommender))
                    np.save(pred_file, y_pred)
                    log("Storing predictions took {} seconds."
                        .format(timedelta(seconds=timer() - t_1)))

                t_1 = timer()
                results = evaluate(self.y_test, y_pred, metrics=self.metrics, batch_size=batch_size)
                log("Evaluation took {} seconds."
                    .format(timedelta(seconds=timer() - t_1)))

                log("\nResults:\n")
                for metric, (mean, std) in zip(self.metrics, results):
                    log("- {}: {} ({})".format(metric, mean, std))
                    log("- {}: {} ({})".format(metric, mean, std), logfile=Path('test.txt'))

                log("\nOverall time: {} seconds."
                    .format(timedelta(seconds=timer() - t_0)))
                log('-' * 79)

    def metrics_calculation(self, rec, model):
        test_csr = self.test_set.tocsr()
        if rec.conditions:
            condition_data_raw = self.test_set.get_attributes(rec.conditions.keys())
            # Important to not call fit here, but just transform
            condition_data = rec.conditions.transform(condition_data_raw)
        else:
            condition_data = None

        y_pred = model.predict(test_csr, condition_data=condition_data)

        #y_pred = model.predict(self.test_set)

        if sp.issparse(y_pred):
            y_pred = y_pred.toarray()
        else:
            # dont hide that we are assuming an ndarray to be returned
            y_pred = np.asarray(y_pred)

        # set likelihood of documents that are already cited to zero, so
        # they don't influence evaluation
        y_pred = remove_non_missing(y_pred, self.x_test, copy=True)
        t_1 = timer()
        results = evaluate(self.y_test, y_pred, metrics=self.metrics, batch_size=10000)
        log("Evaluation took {} seconds."
            .format(timedelta(seconds=timer() - t_1)))

        for metric, (mean, std) in zip(self.metrics, results):
            log("- {}: {} ({})".format(metric, mean, std))


if __name__ == '__main__':
    import doctest

    doctest.testmod()
