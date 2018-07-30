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
from text_match.en.data_utils import datahelper
import pandas as pd
import datetime
import numpy as np

from nltk.tokenize import word_tokenize


def process_data(inputs_data):
    x_test = []
    for line in inputs_data:
        line_list = datahelper.text_to_wordlist(line, remove_stop_words=True, stem_words=False).split(" ")
        x_test.append(line_list)
    return x_test


x_train1, x_train2, _ = datahelper.load_data(filepath_en_train, filepath_sp_train)

x_train1 = process_data(x_train1)
x_train2 = process_data(x_train2)


train = pd.DataFrame()

train['question1'] = x_train1
train['question2'] = x_train2


def total_unique_words(row):
    return len(set(row['question1']).union(row['question2']))


def total_unq_words_stop(row):
    return len([x for x in set(row['question1']).union(row['question2'])])


def wc_diff(row):
    return abs(len(row['question1']) - len(row['question2']))


def wc_ratio(row):
    l1 = len(row['question1']) * 1.0
    l2 = len(row['question2'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def wc_diff_unique(row):
    return abs(len(set(row['question1'])) - len(set(row['question2'])))


def wc_ratio_unique(row):
    l1 = len(set(row['question1'])) * 1.0
    l2 = len(set(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def wc_diff_unique_stop(row):
    return abs(len([x for x in set(row['question1'])]) - len(
        [x for x in set(row['question2'])]))


def wc_ratio_unique_stop(row):
    l1 = len([x for x in set(row['question1'])]) * 1.0
    l2 = len([x for x in set(row['question2'])])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(row['question1'][0] == row['question2'][0])


def char_diff(row):
    return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))


def char_ratio(row):
    l1 = len(''.join(row['question1']))
    l2 = len(''.join(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def char_diff_unique_stop(row):
    return abs(len(''.join([x for x in set(row['question1'])])) - len(
        ''.join([x for x in set(row['question2'])])))


def makeFeature(df_features):
    now = datetime.datetime.now()
    print
    now.strftime('%Y-%m-%d %H:%M:%S')
    df_features['f_total_unique_words'] = df_features.apply(total_unique_words, axis=1, raw=False)
    df_features['f_total_unq_words_stop'] = df_features.apply(total_unq_words_stop, axis=1, raw=False)
    df_features['f_wc_diff'] = df_features.apply(wc_diff, axis=1, raw=False)
    df_features['f_wc_ratio'] = df_features.apply(wc_ratio, axis=1, raw=False)
    df_features['f_wc_diff_unique'] = df_features.apply(wc_diff_unique, axis=1, raw=False)
    df_features['f_wc_ratio_unique'] = df_features.apply(wc_ratio_unique, axis=1, raw=False)
    df_features['f_wc_diff_unique_stop'] = df_features.apply(wc_diff_unique_stop, axis=1, raw=False)
    df_features['f_wc_ratio_unique_stop'] = df_features.apply(wc_ratio_unique_stop, axis=1, raw=False)
    df_features['f_same_start_word'] = df_features.apply(same_start_word, axis=1, raw=False)
    df_features['f_char_diff'] = df_features.apply(char_diff, axis=1, raw=False)
    df_features['f_char_ratio'] = df_features.apply(char_ratio, axis=1, raw=False)
    df_features['f_char_diff_unique_stop'] = df_features.apply(char_diff_unique_stop, axis=1, raw=False)
    now = datetime.datetime.now()
    print
    now.strftime('%Y-%m-%d %H:%M:%S')
    return df_features


if __name__ == "__main__":
    train = train.fillna(' ')

    train = makeFeature(train)
    train.to_csv('train_simple.csv', index=False)
