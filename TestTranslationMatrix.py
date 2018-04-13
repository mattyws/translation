import logging

from gensim.models import TranslationMatrix
from gensim.models.keyedvectors import KeyedVectors
from googletrans import Translator
from sklearn.metrics.classification import accuracy_score, recall_score, precision_score, f1_score

import learn
import os
from helpers import load_obj, save_obj

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model_path = "/home/mattyws/Downloads/Wikipedia/br/fasttext_model/"
model_file = "portuguese_wikipedia.model"

print("================================== Loading models ==================================")
translator = Translator()
word2vecTrainer = learn.Word2VecTrainer()
br_model = word2vecTrainer.load_model(model_path+model_file)
en_model = word2vecTrainer.load_google_model("/home/mattyws/Downloads/Wikipedia/GoogleNews-vectors-negative300.bin")
word_freq = load_obj('word_count')

print("================================== Loading pairs ==================================")
word_pairs = load_obj('word_pairs3')
if os.path.exists('/home/mattyws/Downloads/Wikipedia/br/test_word_pairs.pkl'):
    print("================================== Loading pairs ==================================")
    test_word_pairs = load_obj('test_word_pairs')
else:
    i = 0
    test_word_pairs = []
    while len(test_word_pairs) < 1000 and i < len(word_freq):
        word = word_freq[i][0]
        print(i, len(test_word_pairs), word)
        try:
            translation = translator.translate(word, src='pt', dest='en').text.lower()
            if translation in en_model.wv.vocab and (word, translation) not in word_pairs:
                if not len(translation.split(' ')) > 1:
                    test_word_pairs.append((word, translation))
            i += 1
        except Exception as e:
            i+=1
            print(e)
    save_obj(test_word_pairs, 'test_word_pairs')


print("================================== Testing Translation Matrix ==================================")
translation_model = TranslationMatrix.load("/home/mattyws/Downloads/Wikipedia/translation_matrix.model")
real = []
pred = []
top5_pred = []
for pair in test_word_pairs:
    translation = translation_model.translate(pair[0], sample_num=5).popitem()
    real.append(pair[1])
    pred.append(translation[1][0])
    if pair[1] in translation[1]:
        top5_pred.append(pair[1])
    else:
        top5_pred.append(translation[1][0])

save_obj(pred, 'prediction')
save_obj(real, 'real')
save_obj(top5_pred, 'top_5_prediction')
accuracy = accuracy_score(real, pred)
recall = recall_score(real, pred, average='weighted')
precision = precision_score(real, pred, average='weighted')
f1 = f1_score(real, pred, average='weighted')
print("Accuracy " + str(accuracy), "Recall " + str(recall), "Precision " + str(precision), "F1 " + str(f1))

accuracy = accuracy_score(real, top5_pred)
recall = recall_score(real, top5_pred, average='weighted')
precision = precision_score(real, top5_pred, average='weighted')
f1 = f1_score(real, top5_pred, average='weighted')
print("Top5: Accuracy " + str(accuracy), "Recall " + str(recall), "Precision " + str(precision), "F1 " + str(f1))
