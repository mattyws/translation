import logging

from gensim.models import TranslationMatrix
from gensim.models.keyedvectors import KeyedVectors
from googletrans import Translator

import Learners
import os
from helpers import load_obj, save_obj

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("================================== Loading models ==================================")
translator = Translator()
word2vecTrainer = Learners.FastTextTrainer()
br_model = word2vecTrainer.load_google_model("/home/mattyws/Downloads/Wikipedia/br/wiki.pt/wiki.pt")
en_model = word2vecTrainer.load_google_model("/home/mattyws/Downloads/Wikipedia/wiki.en/wiki.en")
word_freq = load_obj('word_count')

if os.path.exists('/home/mattyws/Downloads/Wikipedia/br/word_pairs_fasttext_inference.pkl'):
    print("================================== Loading pairs ==================================")
    word_pairs = load_obj('word_pairs_fasttext_inference')
else:
    print("================================== Creating pairs ==================================")
    word_pairs = []
    i = 0
    while len(word_pairs) < 5000 and i < len(word_freq):
        word = word_freq[i][0]
        print(i, len(word_pairs), word)
        try:
            translation = translator.translate(word, src='pt', dest='en').text.lower()
            if word in br_model and translation in en_model:
                if not len(translation.split(' ')) > 1:
                    word_pairs.append((word, translation))
            i += 1
        except Exception as e:
            i+=1
            print(e)
    save_obj(word_pairs, 'word_pairs_fasttext_inference')


print("================================== Training Translation Matrix ==================================")
trans_model = TranslationMatrix(br_model.wv, en_model.wv)
trans_model.train(word_pairs)
trans_model.save("/home/mattyws/Downloads/Wikipedia/translation_matrix_fasttext_inference.model")
