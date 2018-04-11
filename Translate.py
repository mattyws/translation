import logging

from gensim.models import TranslationMatrix
from gensim.models.keyedvectors import KeyedVectors
from googletrans import Translator

import learn
import os
from helpers import load_obj, save_obj

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model_path = "/home/mattyws/Downloads/Wikipedia/br/word2vec_model/"
model_file = "portuguese_wikipedia.model"

print("================================== Loading models ==================================")
translator = Translator()
word2vecTrainer = learn.Word2VecTrainer()
br_model = word2vecTrainer.load_model(model_path+model_file)
en_model = word2vecTrainer.load_google_model("/home/mattyws/Downloads/Wikipedia/GoogleNews-vectors-negative300.bin")
print(en_model.wv.vocab['ice cream'])
word_freq = load_obj('word_count')

if os.path.exists('/home/mattyws/Downloads/Wikipedia/br/word_pairs3.pkl'):
    print("================================== Loading pairs ==================================")
    word_pairs = load_obj('word_pairs')
else:
    print("================================== Creating pairs ==================================")
    word_pairs = []
    i = 0
    while len(word_pairs) < 5000 and i < len(word_freq):
        word = word_freq[i][0]
        print(i, len(word_pairs), word)
        try:
            translation = translator.translate(word, src='pt', dest='en').text.lower()
            if translation in en_model.wv.vocab:
                if not len(translation.split(' ')) > 1:
                    word_pairs.append((word, translation))
            i += 1
        except Exception as e:
            i+=1
            print(e)
    save_obj(word_pairs, 'word_pairs3')


print("================================== Training Translation Matrix ==================================")
trans_model = TranslationMatrix(br_model.wv, en_model.wv)
trans_model.train(word_pairs)
trans_model.save("/home/mattyws/Downloads/Wikipedia/translation_matrix.model")

print(trans_model.translate('teste', sample_num=4))
