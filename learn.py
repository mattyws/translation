import csv

from sklearn.model_selection import KFold
from gensim.models import doc2vec
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from keras.utils.np_utils import to_categorical
from sklearn.metrics.classification import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
from gensim.models.callbacks import CallbackAny2Vec


class EpochSaver(CallbackAny2Vec):
     "Callback to save model after every epoch"
     def __init__(self, path_prefix, epochs=0):
         self.path_prefix = path_prefix
         self.epoch = epochs

     def on_epoch_end(self, model):
         output_path = '{}_epoch{}.model'.format(self.path_prefix, self.epoch)
         print("Save model to {}".format(output_path))
         model.save(output_path)
         self.epoch += 1

class Word2VecTrainer(object):
    """
    Perform training and save gensim word2vec
    """

    def __init__(self, min_count=2, size=200, workers=4, window=3, iter=10):
        self.min_count = min_count
        self.size = size
        self.workers = workers
        self.window = window
        self.iter = iter
        self.model = None

    def train(self, corpus, sg=0, callbacks=None):
        self.model = Word2Vec(corpus, callbacks=callbacks, min_count=self.min_count, size=self.size, workers=self.workers, window=self.window, iter=self.iter, sg=sg)

    def save(self, filename):
        self.model.save(filename)

    def get_model(self):
        return self.model

    def load_model(self, filename):
        return Word2Vec.load(filename)

    def load_google_model(self, filename):
        return KeyedVectors.load_word2vec_format(filename, binary=True)

    def retrain(self, model, corpus, sg=0, iter=10, callbacks=None):
        model.train(corpus, total_examples=model.corpus_count, epochs=iter, callbacks=callbacks)
        self.model = model

class FastTextTrainer(object):
    """
    Perform training and save gensim FastText
    """

    def __init__(self, min_count=2, size=200, workers=4, window=3, iter=10):
        self.min_count = min_count
        self.size = size
        self.workers = workers
        self.window = window
        self.iter = iter
        self.model = None

    def train(self, corpus, sg=0, callbacks=None):
        self.model = Word2Vec(corpus, callbacks=callbacks, min_count=self.min_count, size=self.size,
                              workers=self.workers, window=self.window, iter=self.iter, sg=sg)

    def save(self, filename):
        self.model.save(filename)

    def get_model(self):
        return self.model

    def load_model(self, filename):
        return Word2Vec.load(filename)

    def load_google_model(self, filename):
        return KeyedVectors.load_word2vec_format(filename, binary=True)

    def retrain(self, model, corpus, sg=0, iter=10, callbacks=None):
        model.train(corpus, total_examples=model.corpus_count, epochs=iter, callbacks=callbacks)
        self.model = model
