import os

import gensim
import nltk
import learn
import logging
import getopt
import sys
import re

from data import JsonWikipediaDumpReader, DataCleaner, Word2VecDataIter
from learn import EpochSaver


def atoi(text):
    return int(text) if text.isdigit() else text

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def get_last_model(model_path, model_name):
    list = os.listdir(model_path)
    models = []
    for name in list:
        if re.search(model_name+r'_epoch\d+.model$', name):
            models.append(name)
    models.sort(key=natural_keys)
    last_model = models[-1] if len(models)>0 else None
    return last_model

def get_elapsed_epoch(model_name):
    return natural_keys(model_name)[-2]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sg = 1

'''
Configurations
'''
language = 'portuguese'
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words(language)
stemmer = gensim.parsing.PorterStemmer()
output_model_path = "/media/mattyws/Dados/Wikipedia/models/"
output_model_file = "portuguese_wikipedia_fasttext"
iteration = 10
size = 300
last_model = get_last_model(output_model_path, output_model_file)


print("============================= Loading data =============================")
startpath="/media/mattyws/Dados/Wikipedia/text.json"
all_corpus = JsonWikipediaDumpReader(startpath, serve_forever=True)
cleaner = DataCleaner(clean_text=True, tokenizer=tokenizer, stop_set=stop_set)
data = Word2VecDataIter(all_corpus, cleaner)

if last_model is None:
    epoch_saver = EpochSaver(output_model_path+output_model_file)
    print("=============================== Training Model ===============================")
    word2vecTrainer = learn.FastTextTrainer(iter=iteration, size=size)
    word2vecTrainer.train(all_corpus, sg=sg, callbacks=[epoch_saver])
else:
    iteration -= int(get_elapsed_epoch(last_model))
    epoch_saver = EpochSaver(output_model_path+output_model_file, int(get_elapsed_epoch(last_model))+1)
    print("=============================== Training Existing Model ===============================")
    word2vecTrainer = learn.FastTextTrainer(iter=iteration, size=size)
    model = word2vecTrainer.load_model(output_model_path+last_model)
    word2vecTrainer.retrain(model, all_corpus, sg=sg, iter=iteration, callbacks=[epoch_saver])

word2vecTrainer.save(output_model_path+output_model_file+".model")
