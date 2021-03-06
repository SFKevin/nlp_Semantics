import pandas as pd
from CIKM.datautils import datahelper

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

train_bag = pd.read_csv('I:\\temp\\CNNimpl_static\\CIKM\\features\\bags\\train_bagofwords400.csv')
test_bag = pd.read_csv('I:\\temp\\CNNimpl_static\\CIKM\\features\\bags\\test_bagofwords400.csv')
train_bag.drop(['question1', 'question2', 'f_bag_words'], axis=1, inplace=True)
test_bag.drop(['question1', 'question2', 'f_bag_words'], axis=1, inplace=True)

# train_magic1 = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\magic\\new_magic_train.csv")
# test_magic1 = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\magic\\new_magic_test.csv")
# train_magic2 = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\magic\\train_magic2.csv")
# test_magic2 = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\magic\\test_magic2.csv")
# train_freq = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\magic\\train_freq.csv")
# test_freq = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\magic\\test_freq.csv")

train_magic1 = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\magic\\new_magic_train.csv")
test_magic1 = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\magic\\new_magic_test.csv")
train_magic2 = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\magic\\train_magic2.csv")
test_magic2 = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\magic\\test_magic2.csv")
train_freq = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\magic\\train_freq.csv")
test_freq = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\magic\\test_freq.csv")

# train_ngram = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\ngrams\\train_gram_feature.csv")
# test_ngram = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\ngrams\\test_gram_feature.csv")
# train_ngram.drop(['question1', 'question2', 'questions'], inplace=True)
# test_ngram.drop(['question1', 'question2', 'questions'], inplace=True)

train_ngram = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\ngrams\\train_gram_feature.csv")
test_ngram = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\ngrams\\test_gram_feature.csv")
train_ngram.drop(['question1', 'question2', 'questions'], axis=1, inplace=True)
test_ngram.drop(['question1', 'question2', 'questions'], axis=1, inplace=True)

train_simple = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\stringbase\\train_simple.csv")
test_simple = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\stringbase\\test_simple.csv")
train_simple.drop(['question1', 'question2'], axis=1, inplace=True)
test_simple.drop(['question1', 'question2'], axis=1, inplace=True)

# train_weight = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\testunique\\train_weight_noweight.csv")
# test_weight = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\testunique\\test_weight_noweight.csv")
# train_weight.drop(['question1', 'question2', 'q1_unique', 'q2_unique'], inplace=True)
# test_weight.drop(['question1', 'question2', 'q1_unique', 'q2_unique'], inplace=True)

train_weight = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\testunique\\train_weight_noweight.csv")
test_weight = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\testunique\\test_weight_noweight.csv")
train_weight.drop(['question1', 'question2', 'q1_unique', 'q2_unique'], axis=1, inplace=True)
test_weight.drop(['question1', 'question2', 'q1_unique', 'q2_unique'], axis=1, inplace=True)

# train_page = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\pagerank\\pagerank_train.csv")
# test_page = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\pagerank\\pagerank_test.csv")

train_page = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\pagerank\\pagerank_train.csv")
test_page = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\pagerank\\pagerank_test.csv")

# train_w2v = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\word2ve\\train_weight_tfidf.csv")
# test_w2v = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\word2ve\\test_weight_tfidf.csv")
# train_w2v.drop(['question1', 'question2', 'q1_unique', 'q2_unique'], inplace=True)
# test_w2v.drop(['question1', 'question2', 'q1_unique', 'q2_unique'], inplace=True)

train_w2v = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\word2ve\\train_weight_tfidf.csv")
test_w2v = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\word2ve\\test_weight_tfidf.csv")
train_w2v.drop(['question1', 'question2', 'q1_unique', 'q2_unique'], axis=1, inplace=True)
test_w2v.drop(['question1', 'question2', 'q1_unique', 'q2_unique'], axis=1, inplace=True)

# train_doc2vec = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\doc2vec\\train_doc2vec1.csv")
# test_doc2vec = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\doc2vec\\test_doc2vec1.csv")
# train_doc2vec.drop(['doc2vec_train1', 'doc2vec_train2'], inplace=True)
# test_doc2vec.drop(['doc2vec_train1', 'doc2vec_train2'], inplace=True)

train_doc2vec = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\doc2vec\\train_doc2vec1.csv")
test_doc2vec = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\doc2vec\\test_doc2vec1.csv")
train_doc2vec.drop(['doc2vec_train1', 'doc2vec_train2'], axis=1, inplace=True)
test_doc2vec.drop(['doc2vec_test1', 'doc2vec_test2'], axis=1, inplace=True)

train = pd.concat(
    [train_bag, train_magic1, train_magic2, train_freq, train_ngram, train_simple, train_weight, train_page, train_w2v,
     train_doc2vec],
    axis=1)
test = pd.concat(
    [test_bag, test_magic1, test_magic2, test_freq, test_ngram, test_simple, test_weight, test_page, test_w2v,
     test_doc2vec], axis=1)
_, _, _, y_train, _, _ = datahelper.load_data(filepath_en_train, filepath_sp_train)
import numpy as np

sums = np.sum(y_train, axis=0)

print(sums / len(y_train))

from sklearn.model_selection import train_test_split

x_train, x_dev, y_train, y_dev = train_test_split(train.values, y_train, test_size=0.1, random_state=0)
import lightgbm as lgb

train_input = lgb.Dataset(x_train, y_train)

val_input = lgb.Dataset(x_dev, y_dev)

# params = {}
# params["objective"] = "binary:logistic"
# params['eval_metric'] = 'logloss'
# params["eta"] = 0.05
# params["subsample"] = 0.8
# params["min_child_weight"] = 1
# params["colsample_bytree"] = 0.7
# params["max_depth"] = 8
# params["silent"] = 1
# params["seed"] = 1632
# params["gama"] = 0.01

params = {}
params["objective"] = 'binary'
params['metric'] = "binary_logloss"
params["learning_rate"] = 0.1
params["subsample"] = 0.8
params["feature_fraction"] = 0.8
params["max_depth"] = 8
params["num_leaves"] = 256
params["lambda_l1"] = 0.01
params["lambda_l2"] = 0.01
params["num_iterations"] = 500

print("start training")

lgb_model = lgb.train(params=params, train_set=train_input, valid_sets=val_input, early_stopping_rounds=30)

print("predict")
pred = lgb_model.predict(data=test.values, num_iteration=lgb_model.best_iteration)

# a = 0.155 / 0.247
# b = (1 - 0.155) / (1 - 0.247)
#
#
# def func(predictions):
#     for i in range(len(predictions)):
#         x = predictions[i]
#         predictions[i] = a * x / (a * x + b * (1 - x))
#     return predictions
#
#
# pred_weight = func(pred)

np.savetxt("result_lgb20180722.txt", pred)
print("number of result: %d" % len(pred))

# params["min_split_gain"] = 0.01
# d_train = xgb.DMatrix(x_train, label=y_train)
# d_valid = xgb.DMatrix(x_dev, label=y_dev)
# watchlist = [(d_train, 'train'), (d_valid, 'valid')]
# bst = xgb.train(params, d_train, 700, watchlist, early_stopping_rounds=60, verbose_eval=100)  # change to higher #s
# print(log_loss(train.is_duplicate, bst.predict(xgb.DMatrix(train.values))))
#
# sub = pd.DataFrame()
# sub['test_id'] = test['test_id']
# sub['is_duplicate'] = bst.predict(xgb.DMatrix(test.values))
# print
# params
# sub.to_csv('esm_add_pr_400_gamma001.csv', index=False)
