import logging

from gensim.models import TranslationMatrix
from gensim.models.keyedvectors import KeyedVectors
from googletrans import Translator

import learn
import os
from helpers import load_obj, save_obj

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("================================== Loading models ==================================")
translator = Translator()
word2vecTrainer = learn.FastTextTrainer()
br_model = word2vecTrainer.load_google_model("/home/mattyws/Downloads/Wikipedia/wiki.pt/wiki.pt")
word_freq = load_obj('word_count')

print("================================== Creating pairs ==================================")
i = 0
not_in_vocab = set()
infer_words = set()
while i < len(word_freq):
    word = word_freq[i][0]
    try:
        if word not in br_model.wv.vocab:
            not_in_vocab.add(word)
        if word in br_model and word not in br_model.wv.vocab:
            infer_words.add(word)
        i += 1
    except Exception as e:
        i+=1
        print(e)


print("Words infered", infer_words)
print("Words that could not be infered", not_in_vocab.difference(infer_words))