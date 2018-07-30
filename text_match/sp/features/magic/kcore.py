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
from CIKM.datautils import datahelper
import pandas as pd
from collections import defaultdict


def process_data(inputs_data):
    stop_word = list(open(file_stop_word, "r", encoding='UTF-8').readlines())
    stop_word_list = [
        line.replace("\n", "").replace(",", "").replace(".", "").replace("?", "").replace("¿", "").replace("!",
                                                                                                           "").replace(
            "¡", "").lower() for
        line in
        stop_word]
    x_test = []
    for line in inputs_data:
        line_list = line.split(" ")
        line_list = [x for x in line_list if x not in stop_word_list]
        x_test.append(line_list)
    return x_test


x_train1, x_train2, _, _, _, _ = datahelper.load_data(filepath_en_train, filepath_sp_train)
x_test1, x_test2 = datahelper.load_testdata(filepath_test)

# x_train1 = process_data(x_train1)
# x_train2 = process_data(x_train2)
# x_test1 = process_data(x_test1)
# x_test2 = process_data(x_test2)

train = pd.DataFrame()
test = pd.DataFrame()

train['question1'] = x_train1
train['question2'] = x_train2

test['question1'] = x_test1
test['question2'] = x_test2

ques = pd.concat([train[['question1', 'question2']], test[['question1', 'question2']]], axis=0).reset_index(
    drop='index')

q_dict = defaultdict(set)
for i in range(ques.shape[0]):
    q_dict[ques.question1[i]].add(ques.question2[i])
    q_dict[ques.question2[i]].add(ques.question1[i])


def q1_q2_intersect(row):
    return (len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))


train_orig = pd.DataFrame()
test_orig = pd.DataFrame()

train_orig['z_q1_q2_intersect'] = train.apply(q1_q2_intersect, axis=1, raw=False)
test_orig['z_q1_q2_intersect'] = test.apply(q1_q2_intersect, axis=1, raw=False)

# train_orig['z_q1_q2_intersect'] = train.apply(q1_q2_intersect, axis=1, raw=False)
# test_orig['z_q1_q2_intersect'] = test.apply(q1_q2_intersect, axis=1, raw=False)

train_orig.to_csv('train_magic2.csv', index=False)
test_orig.to_csv('test_magic2.csv', index=False)
