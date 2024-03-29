""" Denoising Autoencoders """
# CFR https://gist.github.com/bigsnarfdude/dde651f6e06f266b48bc3750ac730f80,
# https://github.com/GunhoChoi/Kind-PyTorch-Tutorial/tree/master/07_Denoising_Autoencoder

# torch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.autograd import Variable

# sklearn
import sklearn

# numpy
import numpy as np

# own recommender stuff
from models.base import Recommender
from models.datasets import Bags
from models.evaluation import Evaluation
from gensim.models.keyedvectors import KeyedVectors

from models.condition import ConditionList, _check_conditions, PretrainedWordEmbeddingCondition

torch.manual_seed(42)
TINY = 1e-12

STATUS_FORMAT = "[ R: {:.4f} | D: {:.4f} | G: {:.4f} ]"


def log_losses(*losses):
    print('\r' + STATUS_FORMAT.format(*losses), end='', flush=True)


def gauss_noise(batch, noise_factor):
    '''Add gaussian noise to the input'''
    noise = torch.randn(batch.size()) * noise_factor
    if torch.cuda.is_available():
        noise = noise.cuda()
    return (batch + noise)


def zeros_noise(batch, noise_factor):
    '''Randomly zeros some of the 1s with p=noise_factor'''
    mask = torch.rand(batch.size()) < noise_factor
    batch[mask] = 0
    return batch

TORCH_OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam
}

NOISE_TYPES ={
    'gauss': gauss_noise,
    'zeros': zeros_noise
}


class Encoder(nn.Module):
    """ Three-layer Encoder """

    def __init__(self, n_input, n_hidden, n_code, final_activation=None,
                 normalize_inputs=True, dropout=(.2, .2), activation='ReLU'):
        super(Encoder, self).__init__()
        self.lin1 = nn.Linear(n_input, n_hidden)
        self.act1 = getattr(nn, activation)()
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.act2 = getattr(nn, activation)()
        if activation == 'SELU':
            self.drop1 = nn.AlphaDropout(dropout[0])
            self.drop2 = nn.AlphaDropout(dropout[1])
        else:
            self.drop1 = nn.Dropout(dropout[0])
            self.drop2 = nn.Dropout(dropout[1])
        self.lin3 = nn.Linear(n_hidden, n_code)
        self.normalize_inputs = normalize_inputs
        if final_activation == 'linear' or final_activation is None:
            self.final_activation = None
        elif final_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            raise ValueError("Final activation unknown:", activation)

    def forward(self, inp):
        """ Forward method implementation of 3-layer encoder """
        if self.normalize_inputs:
            inp = F.normalize(inp, 1)
        # first layer
        act = self.lin1(inp)
        act = self.drop1(act)
        act = self.act1(act)
        # second layer
        act = self.lin2(act)
        act = self.drop2(act)
        act = self.act2(act)
        # third layer
        act = self.lin3(act)
        if self.final_activation:
            act = self.final_activation(act)
        return act


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, n_code, n_hidden, n_output, dropout=(.2, .2), activation='ReLU'):
        super(Decoder, self).__init__()
        self.lin1 = nn.Linear(n_code, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, n_output)
        if activation == 'SELU':
            self.drop1 = nn.AlphaDropout(dropout[0])
            self.drop2 = nn.AlphaDropout(dropout[1])
        else:
            self.drop1 = nn.Dropout(dropout[0])
            self.drop2 = nn.Dropout(dropout[1])
        self.act1 = getattr(nn, activation)()
        self.act2 = getattr(nn, activation)()

    def forward(self, inp):
        """ Forward implementation of 3-layer decoder """
        # first layer
        act = self.lin1(inp)
        act = self.drop1(act)
        act = self.act1(act)
        # second layer
        act = self.lin2(act)
        act = self.drop2(act)
        act = self.act2(act)
        # final layer
        act = self.lin3(act)
        act = F.sigmoid(act)
        return act


class DenoisingAutoEncoder():
    def __init__(self,
                 n_hidden=100,
                 n_code=50,
                 lr=0.001,
                 batch_size=100,
                 n_epochs=500,
                 optimizer='adam',
                 normalize_inputs=True,
                 activation='ReLU',
                 dropout=(.2, .2),
                 noise_factor=0.2,
                 corrupt='zeros',
                 conditions=None,
                 verbose=True):

        self.enc, self.dec = None, None
        self.n_hidden = n_hidden
        self.n_code = n_code
        self.n_epochs = n_epochs
        self.optimizer = optimizer.lower()
        self.normalize_inputs = normalize_inputs
        self.verbose = verbose
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.activation = activation
        self.noise_factor = noise_factor
        self.corrupt = NOISE_TYPES[corrupt.lower()]
        self.conditions = conditions

    def eval(self):
        """ Put all NN modules into eval mode """
        self.enc.eval()
        self.dec.eval()
        if self.conditions:
            self.conditions.eval()

    def train(self):
        """ Put all NN modules into train mode """
        self.enc.train()
        self.dec.train()
        if self.conditions:
            self.conditions.train()

    def ae_step(self, batch, condition_data=None):
        """ Perform one autoencoder training step """
        z_sample = self.enc(self.corrupt(batch, self.noise_factor))

        use_condition = _check_conditions(self.conditions, condition_data)
        if use_condition:
            z_sample = self.conditions.encode_impose(z_sample, condition_data)

        x_sample = self.dec(z_sample)
        recon_loss = F.binary_cross_entropy(x_sample + TINY,
                                            batch.view(batch.size(0),
                                                       batch.size(1)) + TINY)
        self.enc_optim.zero_grad()
        self.dec_optim.zero_grad()
        if self.conditions:
            self.conditions.zero_grad()
        recon_loss.backward()
        self.enc_optim.step()
        self.dec_optim.step()
        if self.conditions:
            self.conditions.step()
        return recon_loss.item()

    def partial_fit(self, X, y=None, condition_data=None):
        """ Performs reconstrction, discimination, generator training steps """
        _check_conditions(self.conditions, condition_data)

        if y is not None:
            raise ValueError("(Semi-)supervised usage not supported")
        # Transform to Torch (Cuda) Variable, shift batch to GPU
        X = Variable(torch.FloatTensor(X))
        if torch.cuda.is_available():
            X = X.cuda()


        # Make sure we are in training mode and zero leftover gradients
        self.train()
        # One step each, could balance
        recon_loss = self.ae_step(X, condition_data=condition_data)
        if self.verbose:
            log_losses(recon_loss, 0, 0)
        return self

    def fit(self, X, y=None, condition_data=None):
        if y is not None:
            raise NotImplementedError("(Semi-)supervised usage not supported")

        use_condition = _check_conditions(self.conditions, condition_data)

        if use_condition:
            code_size = self.n_code + self.conditions.size_increment()
        else:
            code_size = self.n_code

        self.enc = Encoder(X.shape[1], self.n_hidden, self.n_code,
                           final_activation='linear',
                           normalize_inputs=self.normalize_inputs,
                           dropout=self.dropout, activation=self.activation)
        self.dec = Decoder(code_size, self.n_hidden,
                           X.shape[1], dropout=self.dropout, activation=self.activation)

        if torch.cuda.is_available():
            self.enc = self.enc.cuda()
            self.dec = self.dec.cuda()
        optimizer_gen = TORCH_OPTIMIZERS[self.optimizer]
        # Reconstruction
        self.enc_optim = optimizer_gen(self.enc.parameters(), lr=self.lr)
        self.dec_optim = optimizer_gen(self.dec.parameters(), lr=self.lr)

        # do the actual training
        for epoch in range(self.n_epochs):
            if self.verbose:
                print("Epoch", epoch + 1)

            # Shuffle on each new epoch
            if use_condition:
                # shuffle(*arrays) takes several arrays and shuffles them so indices are still matching
                X_shuf, *condition_data_shuf = sklearn.utils.shuffle(X, *condition_data)
            else:
                X_shuf = sklearn.utils.shuffle(X)

            for start in range(0, X.shape[0], self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuf[start:end].toarray()
                # condition may be None
                if use_condition:
                    # c_batch = condition_shuf[start:(start+self.batch_size)]
                    c_batch = [c[start:end] for c in condition_data_shuf]
                    self.partial_fit(X_batch, condition_data=c_batch)
                else:
                    self.partial_fit(X_batch)

            if self.verbose:
                # Clean up after flushing batch loss printings
                print()
        return self

    def predict(self, X, condition_data=None):
        use_condition = _check_conditions(self.conditions, condition_data)
        self.eval()  # Deactivate dropout
        if self.conditions:
            self.conditions.eval()
        pred = []

        with torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                # batched predictions, yet inclusive
                end = start + self.batch_size
                X_batch = X[start:end].toarray()
                X_batch = torch.FloatTensor(X_batch)
                if torch.cuda.is_available():
                    X_batch = X_batch.cuda()
                X_batch = Variable(X_batch)

                if use_condition:
                    c_batch = [c[start:end] for c in condition_data]

                z = self.enc(X_batch)
                if use_condition:
                    z = self.conditions.encode_impose(z, c_batch)
                # reconstruct
                X_reconstuction = self.dec(z)
                # shift
                X_reconstuction = X_reconstuction.data.cpu().numpy()
                pred.append(X_reconstuction)
        return np.vstack(pred)


class DAERecommender(Recommender):
    """
    Denoising Recommender
    =====================================

    Arguments
    ---------
    n_input: Dimension of input to expect
    n_hidden: Dimension for hidden layers
    n_code: Code Dimension

    Keyword Arguments
    -----------------
    n_epochs: Number of epochs to train
    batch_size: Batch size to use for training
    verbose: Print losses during training
    normalize_inputs: Whether l1-normalization is performed on the input
    """

    def __init__(self, conditions=None, **kwargs):
        """ tfidf_params get piped to either TfidfVectorizer or
        EmbeddedVectorizer.  Remaining kwargs get passed to
        AdversarialAutoencoder """
        super().__init__()
        self.verbose = kwargs.get('verbose', True)
        self.model_params = kwargs
        self.conditions = conditions
        self.dae = None

    def __str__(self):
        desc = "Denoising Autoencoder"
        if self.conditions:
            desc += " conditioned on: " + ', '.join(self.conditions.keys())
        desc += '\nDAE Params: ' + str(self.model_params)

        return desc

    def train(self, training_set):
        X = training_set.tocsr()
        if self.conditions:
            condition_data_raw = training_set.get_attributes(self.conditions.keys())
            condition_data = self.conditions.fit_transform(condition_data_raw)
        else:
            condition_data = None

        self.dae = DenoisingAutoEncoder(conditions=self.conditions, **self.model_params)

        print(self)
        print(self.dae)
        print(self.conditions)

        self.dae.fit(X, condition_data=condition_data)

    def predict(self, test_set):
        X = test_set.tocsr()
        if self.conditions:
            condition_data_raw = test_set.get_attributes(self.conditions.keys())
            # Important to not call fit here, but just transform
            condition_data = self.conditions.transform(condition_data_raw)
        else:
            condition_data = None

        pred = self.dae.predict(X, condition_data=condition_data)

        return pred
