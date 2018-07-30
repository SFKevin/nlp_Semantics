import pandas as pd
from text_match.en.data_utils import datahelper

# filepath_en_train = "E:\\CIKM2018\\cikm_english_train_20180516\\cikm_english_train_20180516.txt"
# filepath_sp_train = "E:\\CIKM2018\\cikm_spanish_train_20180516.txt"
# filepath_test = "E:\\CIKM2018\\cikm_test_a_20180516.txt"
# filepath_unlabel = "E:\\CIKM2018\\cikm_unlabel_spanish_train_20180516\\cikm_unlabel_spanish_train_20180516.txt"
# w2v_pah = "E:\\CIKM2018\\w2v.model.bin"
# fast_path = "E:\\CIKM2018\\fast_text_vectors_wiki.es.vec\\wiki.es.vec"
# file_stop_word = "E:\\CIKM2018\\spanish_stop_word.txt"
filepath_en_train = "I:\\CIKM\\cikm_english_train_20180516\\cikm_english_train_20180516.txt"
filepath_sp_train = "I:\\CIKM\\cikm_spanish_train_20180516.txt"
filepath_test = "I:\\CIKM\\cikm_test_a_20180516.txt"
filepath_unlabel = "I:\\CIKM\\cikm_unlabel_spanish_train_20180516\\cikm_unlabel_spanish_train_20180516.txt"
w2v_pah = "I:\\CIKM\\w2v.model.bin"
fast_path = "I:\\CIKM\\fast_text_vectors_wiki.es.vec\\wiki.es.vec"
file_stop_word = "I:\\CIKM\\spanish_stop_word.txt"

train_bag = pd.read_csv('I:\\nlp_semantics\\text_match\\en\\features\\bags\\train_bagofwords400.csv')
train_bag.drop(['question1', 'question2', 'f_bag_words'], axis=1, inplace=True)

train_magic1 = pd.read_csv("I:\\nlp_semantics\\text_match\\en\\features\\magic\\new_magic_train.csv")
train_magic2 = pd.read_csv("I:\\nlp_semantics\\text_match\\en\\features\\magic\\train_magic2.csv")
train_freq = pd.read_csv("I:\\nlp_semantics\\text_match\\en\\features\\magic\\train_freq.csv")

train_ngram = pd.read_csv("I:\\nlp_semantics\\text_match\\en\\features\\ngrams\\train_gram_feature.csv")
train_ngram.drop(['question1', 'question2', 'questions'], axis=1, inplace=True)

train_simple = pd.read_csv("I:\\nlp_semantics\\text_match\\en\\features\\stringbase\\train_simple.csv")
train_simple.drop(['question1', 'question2'], axis=1, inplace=True)

train_weight = pd.read_csv("I:\\nlp_semantics\\text_match\\en\\features\\testunique\\train_weight_noweight.csv")
train_weight.drop(['question1', 'question2', 'q1_unique', 'q2_unique'], axis=1, inplace=True)

train_page = pd.read_csv("I:\\nlp_semantics\\text_match\\en\\features\\pagerank\\pagerank_train.csv")

train_w2v = pd.read_csv("I:\\nlp_semantics\\text_match\\en\\features\\word2ve\\train_weight_tfidf.csv")
train_w2v.drop(['question1', 'question2', 'q1_unique', 'q2_unique'], axis=1, inplace=True)

train_doc2vec = pd.read_csv("I:\\nlp_semantics\\text_match\\en\\features\\doc2vec\\train_doc2vec1.csv")
train_doc2vec.drop(['doc2vec_train1', 'doc2vec_train2'], axis=1, inplace=True)

train = pd.concat(
    [train_bag, train_magic1, train_magic2, train_freq, train_ngram, train_simple, train_weight, train_page, train_w2v,
     train_doc2vec],
    axis=1)
_, _, y_train = datahelper.load_data(filepath_en_train, filepath_sp_train)
import numpy as np

sums = np.sum(y_train, axis=0)

print(sums / len(y_train))

from sklearn.model_selection import train_test_split

x_train, x_dev, y_train, y_dev = train_test_split(train.values, y_train, test_size=0.1, random_state=0)
import lightgbm as lgb

train_input = lgb.Dataset(x_train, y_train)

val_input = lgb.Dataset(x_dev, y_dev)

params = {}
params["objective"] = 'binary'
params['metric'] = "binary_logloss"
params["learning_rate"] = 0.01
params["subsample"] = 0.8
params["feature_fraction"] = 0.8
params["max_depth"] = 8
params["num_leaves"] = 256
params["lambda_l1"] = 0.1
params["lambda_l2"] = 0.1
params["num_iterations"] = 3000

print("start training")

lgb_model = lgb.train(params=params, train_set=train_input, valid_sets=val_input)
