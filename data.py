import json
import os
# import ijson
import PreProcess


class DataCleaner:

    def __init__(self, clean_text=False, remove_digits=False, tokenizer=None, stop_set=None, stemmer=None):
        self.clean_text = clean_text
        self.tokenizer = tokenizer
        self.stop_set = stop_set
        self.stemmer = stemmer
        self.remove_digits = remove_digits

    def clean(self, text):
        text = PreProcess.Tokenize(self.tokenizer).tokenize(text)
        if self.stop_set is not None:
            text = PreProcess.CleanStopWords(self.stop_set).clean(text)
        if self.stemmer is not None:
            text = [self.stemmer.stem(word) for word in text]
        if self.remove_digits is True:
            text = [word for word in text if not word.isdigit()]
        return text

class Word2VecDataIter:

    def __init__(self, iterator, cleaner):
        self.iterator = iterator
        self.cleaner = cleaner

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.iterator)
        except:
            raise StopIteration()
        return self.cleaner.clean(data)

class DirWalk:
    """
    A class that walks in a root directory and return the path of the file
    """
    def __init__(self, path, serve_forever=False):
        self.path_list = []
        for root, dirs, files in os.walk(path):
            for file in files:
                self.path_list.append(root+"/"+file)
        self.len = len(self.path_list)
        self.i = -1 #to be eq to 0 after the first next
        self.serve_forever = serve_forever

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.len:
            if self.serve_forever:
                self.i = -1
            else:
                raise StopIteration()
        self.i += 1
        return self.path_list[self.i]

    def __len__(self):
        return self.len

class JsonWikipediaDumpReader:
    """
    Reads a json file generated by the WikiExtractor.py, and returns the text of the content iteratively.
    The file can be the concatenation of files generated by the extractor.
    """
    def __init__(self, path, serve_forever=False):
        """
        Init method
        :param path: the path to the json files
        :param serve_forever: if the class will loop like a circular list
        """
        self.path = path
        self.json_file = open(path, "r")
        self.serve_forever = serve_forever

    def __open_file(self):
        if not self.json_file.closed:
            self.json_file.close()
        self.json_file = open(self.path, "r")

    def __iter__(self):
        return self

    def __next__(self):
        try:
            json_text = next(self.json_file)
            data = json.loads(json_text)
            return data['text']
        except:
            self.__open_file()
            raise StopIteration()
