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

train_rnn_freq = train_freq[['q1_freq', 'q2_freq']]
test_rnn_freq = test_freq[['q1_freq', 'q2_freq']]

train_rnn_magic1 = train_magic1[['q1_q2_intersect', 'q1_q2_wm_ratio']]
test_rnn_magic1 = test_magic1[['q1_q2_intersect', 'q1_q2_wm_ratio']]

train_rnn_magic2 = train_magic2['z_q1_q2_intersect']
test_rnn_magic2 = test_magic2['z_q1_q2_intersect']

# train_ngram = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\ngrams\\train_gram_feature.csv")
# test_ngram = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\ngrams\\test_gram_feature.csv")
# train_ngram.drop(['question1', 'question2', 'questions'], inplace=True)
# test_ngram.drop(['question1', 'question2', 'questions'], inplace=True)

# train_ngram = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\ngrams\\train_gram_feature.csv")
# test_ngram = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\ngrams\\test_gram_feature.csv")
# train_ngram.drop(['question1', 'question2', 'questions'], axis=1, inplace=True)
# test_ngram.drop(['question1', 'question2', 'questions'], axis=1, inplace=True)

train_simple = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\stringbase\\train_simple.csv")
test_simple = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\stringbase\\test_simple.csv")
train_simple.drop(['question1', 'question2'], axis=1, inplace=True)
test_simple.drop(['question1', 'question2'], axis=1, inplace=True)

# train_weight = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\testunique\\train_weight_noweight.csv")
# test_weight = pd.read_csv("E:\\tempwork\\CNNimpl_static\\CIKM\\features\\testunique\\test_weight_noweight.csv")
# train_weight.drop(['question1', 'question2', 'q1_unique', 'q2_unique'], inplace=True)
# test_weight.drop(['question1', 'question2', 'q1_unique', 'q2_unique'], inplace=True)

# train_weight = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\testunique\\train_weight_noweight.csv")
# test_weight = pd.read_csv("I:\\temp\\CNNimpl_static\\CIKM\\features\\testunique\\test_weight_noweight.csv")
# train_weight.drop(['question1', 'question2', 'q1_unique', 'q2_unique'], axis=1, inplace=True)
# test_weight.drop(['question1', 'question2', 'q1_unique', 'q2_unique'], axis=1, inplace=True)

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
    [train_rnn_freq, train_rnn_magic1, train_rnn_magic2, train_simple, train_page, train_w2v,
     train_doc2vec],
    axis=1)
test = pd.concat(
    [test_rnn_freq, test_rnn_magic1, test_rnn_magic2, test_simple, test_page, test_w2v,
     test_doc2vec], axis=1)

train.to_csv("rnn_train.csv", index=False)
test.to_csv("rnn_test.csv", index=False)

# _, _, _, y_train, _, _ = datahelper.load_data(filepath_en_train, filepath_sp_train)
# import numpy as np
#
# sums = np.sum(y_train, axis=0)
#
# print(sums / len(y_train))
#
# from sklearn.model_selection import train_test_split
#
# x_train, x_dev, y_train, y_dev = train_test_split(train.values, y_train, test_size=0.1, random_state=0)
# import lightgbm as lgb
#
# train_input = lgb.Dataset(x_train, y_train)
#
# val_input = lgb.Dataset(x_dev, y_dev)

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

# params = {}
# params["objective"] = 'binary'
# params['metric'] = "binary_logloss"
# params["learning_rate"] = 0.1
# params["subsample"] = 0.8
# params["feature_fraction"] = 0.8
# params["max_depth"] = 8
# params["num_leaves"] = 256
# params["lambda_l1"] = 0.01
# params["lambda_l2"] = 0.01
# params["num_iterations"] = 500
#
# print("start training")
#
# lgb_model = lgb.train(params=params, train_set=train_input, valid_sets=val_input)
#
# print("predict")
# pred = lgb_model.predict(data=test.values, num_iteration=lgb_model.best_iteration)
#
# np.savetxt("result_lgb20180722.txt", pred)
# print("number of result: %d" % len(pred))
