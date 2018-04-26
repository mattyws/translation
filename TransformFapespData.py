import nltk
import numpy as np

from Embedding_Adapters import FastTextAdapter
from data import DataCleaner
'''
Transform the Fapesp or any other parallel data in the same format to a word embedding matrix.
The embedding model can be changed if it has the __getitem__ method implemented (consider using Adapters). 
'''


# Read and return the parallel data into a variable
def create_data(pt_file, en_file, pt_cleaner, en_cleaner):
    pt, en = open(pt_file, 'r'), open(en_file, 'r')
    data = []
    for pt_line, en_line in zip(pt, en):
        pt_newline = pt_cleaner.clean(pt_line)
        en_newline = en_cleaner.clean(en_line)
        data.append([pt_newline, en_newline])
    return np.array(data)


# Transform a list of text into lists of word embeddings
def transform (data, embedding_model):
    new_data = []
    for phrase in data:
        phrase_matrix = []
        for word in phrase:
            phrase_matrix.append(embedding_model[word])
        new_data.append(phrase_matrix)
    return np.array(new_data)



pt_fapesp_train_path = "/home/mattyws/Downloads/fapesp-corpora/pt-en.split/fapesp-v2.pt-en.dev.pt"
en_fapesp_train_path = "/home/mattyws/Downloads/fapesp-corpora/pt-en.split/fapesp-v2.pt-en.dev.en"
pt_fapesp_test_path = "/home/mattyws/Downloads/fapesp-corpora/pt-en.split/fapesp-v2.pt-en.dev-test.pt"
en_fapesp_test_path = "/home/mattyws/Downloads/fapesp-corpora/pt-en.split/fapesp-v2.pt-en.dev-test.en"
pt_embedding_model_path = "/home/mattyws/Downloads/Wikipedia/wiki.pt/wiki.pt"
en_embedding_model_path = "/home/mattyws/Downloads/Wikipedia/wiki.en/wiki.en"


tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_set = nltk.corpus.stopwords.words("portuguese")
cleaner = DataCleaner(clean_text=True, tokenizer=tokenizer, stop_set=stop_set)

print("============================== Loading data ==============================")
data = create_data(pt_fapesp_test_path, en_fapesp_test_path, cleaner, cleaner)
np.random.shuffle(data)

print("============================== Loading Model ==============================")
pt_embedding_model = FastTextAdapter()
pt_embedding_model.load_binary_model(pt_embedding_model_path)
en_embedding_model = FastTextAdapter()
en_embedding_model.load_binary_model(en_embedding_model_path)

print("============================== Transforming Data ==============================")
pt_data = transform(data[:, 0], pt_embedding_model)
en_data = transform(data[:, 1], pt_embedding_model)


