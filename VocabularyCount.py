import pickle

import gensim
import nltk

from data import JsonWikipediaDumpReader, DataCleaner, Word2VecDataIter
import operator

from helpers import save_obj

language = 'portuguese'
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words(language)
stemmer = gensim.parsing.PorterStemmer()
print("============================= Loading data =============================")
startpath = "/home/mattyws/Downloads/Wikipedia/br/data/text.json"
all_corpus = JsonWikipediaDumpReader(startpath, serve_forever=True)
cleaner = DataCleaner(clean_text=True, tokenizer=tokenizer, stop_set=stop_set)
data = Word2VecDataIter(all_corpus, cleaner)

word_count = dict()
for doc in data:
    for word in doc:
        if word not in word_count.keys():
            word_count[word] = 0
        word_count[word] += 1

sorted_x = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)
save_obj(sorted_x, 'word_count')
for i in range(0, 5000):
    print(sorted_x[i])
