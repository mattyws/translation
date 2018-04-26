import abc

from gensim.models import FastText


class EmbeddingAdapter(object):

    __mataclass__ = abc.ABCMeta

    model = None

    @abc.abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError('users must define \'__getitem__\' to use this base class')

    def get_model(self):
        return self.model

    @abc.abstractmethod
    def load_model(self, filename):
        raise NotImplementedError('users must define \'load_model\' to use this base class')

    @abc.abstractmethod
    def load_binary_model(self, filename):
        raise NotImplementedError('users must define \'load_binary_model\' to use this base class')


class FastTextAdapter(EmbeddingAdapter):

    def load_model(self, filename):
        self.model = FastText.load(filename)

    def load_binary_model(self, filename):
        self.model = FastText.load_fasttext_format(filename)

    def __getitem__(self, item):
        return self.model[item]