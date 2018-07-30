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
from collections import defaultdict
from CIKM.datautils import datahelper
import pandas as pd

stop_word = list(open(file_stop_word, "r", encoding='UTF-8').readlines())
stop_word_list = [
    line.replace("\n", "").replace(",", "").replace(".", "").replace("?", "").replace("¿", "").replace("!",
                                                                                                       "").replace(
        "¡", "").lower() for
    line in
    stop_word]


def word_match_share(q1, q2):
    q1 = str(q1).lower().split()
    q2 = str(q2).lower().split()
    q1words = {}
    q2words = {}
    for word in q1:
        if word not in stop_word_list:
            q1words[word] = 1
    for word in q2:
        if word not in stop_word_list:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0.
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    return R


x_train1, x_train2, _, _, _, _ = datahelper.load_data(filepath_en_train, filepath_sp_train)
x_test1, x_test2 = datahelper.load_testdata(filepath_test)

ques = pd.DataFrame()

ques['question1'] = x_train1 + x_test1
ques['question2'] = x_train2 + x_test2

train = pd.DataFrame()
test = pd.DataFrame()

train['question1'] = x_train1
train['question2'] = x_train2

test['question1'] = x_test1
test['question2'] = x_test2

# train.to_csv("train.csv")
# train.to_csv("test.csv")

q_dict = defaultdict(dict)
for i in range(ques.shape[0]):
    wm = word_match_share(ques.question1[i], ques.question2[i])
    q_dict[ques.question1[i]][ques.question2[i]] = wm
    q_dict[ques.question2[i]][ques.question1[i]] = wm


def q1_q2_intersect(row):
    return (len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))


def q1_q2_wm_ratio(row):
    q1 = q_dict[row[0]]
    q2 = q_dict[row[1]]
    inter_keys = set(q1.keys()).intersection(set(q2.keys()))
    if (len(inter_keys) == 0): return 0.
    inter_wm = 0.
    total_wm = 0.
    for q, wm in q1.items():
        if q in inter_keys:
            inter_wm += wm
        total_wm += wm
    for q, wm in q2.items():
        if q in inter_keys:
            inter_wm += wm
        total_wm += wm
    if (total_wm == 0.):
        return 0.
    return inter_wm / total_wm


train['q1_q2_wm_ratio'] = train.apply(q1_q2_wm_ratio, axis=1, raw=True)
test['q1_q2_wm_ratio'] = test.apply(q1_q2_wm_ratio, axis=1, raw=True)

train['q1_q2_intersect'] = train.apply(q1_q2_intersect, axis=1, raw=True)
test['q1_q2_intersect'] = test.apply(q1_q2_intersect, axis=1, raw=True)

train_feat = train[['q1_q2_intersect', 'q1_q2_wm_ratio']]
test_feat = test[['q1_q2_intersect', 'q1_q2_wm_ratio']]

train_feat.to_csv('new_magic_train.csv', index=False)
test_feat.to_csv('new_magic_test.csv', index=False)
