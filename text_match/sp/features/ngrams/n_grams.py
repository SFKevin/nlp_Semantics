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
import numpy as np
from nltk import ngrams
from simhash import Simhash
import datetime
from multiprocessing import Pool


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

from nltk.tokenize import word_tokenize


def tokenize(sequence):
    words = word_tokenize(sequence)
    return words


def Jaccarc(q1words, q2words):
    if isinstance(q1words, str):
        q1words = q1words.split()
        q2words = q2words.split()
    tot = len(q1words) + len(q2words) + 1
    same = 0
    for word1 in q1words:
        for word2 in q2words:
            if (word1 == word2):
                same += 1
    return float(same) / (float(tot - same) + 0.000001)


def Dice(q1words, q2words):
    if isinstance(q1words, str):
        q1words = q1words.split()
        q2words = q2words.split()
    tot = len(q1words) + len(q2words) + 1
    same = 0
    for word1 in q1words:
        for word2 in q2words:
            if (word1 == word2):
                same += 1
    return 2 * float(same) / (float(tot) + 0.000001)


def Ochiai(q1words, q2words):
    if isinstance(q1words, str):
        q1words = q1words.split()
        q2words = q2words.split()
    tot = len(q1words) * len(q2words) + 1
    same = 0
    for word1 in q1words:
        for word2 in q2words:
            if (word1 == word2):
                same += 1
    return float(same) / (np.sqrt(float(tot)) + 0.000001)


def get_word_ngrams(sequence, n=3):
    tokens = tokenize(sequence)
    return [' '.join(ngram) for ngram in ngrams(tokens, n)]


def get_character_ngrams(sequence, n=3):
    return [sequence[i:i + n] for i in range(len(sequence) - n + 1)]


def caluclate_simhash_distance(sequence1, sequence2):
    return Simhash(sequence1).distance(Simhash(sequence2))


def get_word_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = tokenize(q1), tokenize(q2)
    return caluclate_simhash_distance(q1, q2)


def get_word_2gram_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
    return caluclate_simhash_distance(q1, q2)


def get_char_2gram_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
    return caluclate_simhash_distance(q1, q2)


def get_word_3gram_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
    return caluclate_simhash_distance(q1, q2)


def get_char_3gram_distance(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
    return caluclate_simhash_distance(q1, q2)


def get_word_distance2(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = tokenize(q1), tokenize(q2)
    return Jaccarc(q1, q2)


def get_word_2gram_distance2(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
    return Jaccarc(q1, q2)


def get_char_2gram_distance2(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
    return Jaccarc(q1, q2)


def get_word_3gram_distance2(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
    return Jaccarc(q1, q2)


def get_char_3gram_distance2(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
    return Jaccarc(q1, q2)


def get_word_distance3(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = tokenize(q1), tokenize(q2)
    return Dice(q1, q2)


def get_word_2gram_distance3(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
    return Dice(q1, q2)


def get_char_2gram_distance3(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
    return Dice(q1, q2)


def get_word_3gram_distance3(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
    return Dice(q1, q2)


def get_char_3gram_distance3(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
    return Dice(q1, q2)


def get_word_distance4(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = tokenize(q1), tokenize(q2)
    return Ochiai(q1, q2)


def get_word_2gram_distance4(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 2), get_word_ngrams(q2, 2)
    return Ochiai(q1, q2)


def get_char_2gram_distance4(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 2), get_character_ngrams(q2, 2)
    return Ochiai(q1, q2)


def get_word_3gram_distance4(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_word_ngrams(q1, 3), get_word_ngrams(q2, 3)
    return Ochiai(q1, q2)


def get_char_3gram_distance4(questions):
    q1, q2 = questions.split('_split_tag_')
    q1, q2 = get_character_ngrams(q1, 3), get_character_ngrams(q2, 3)
    return Ochiai(q1, q2)


def makeFeature(df_features):
    pool = Pool(processes=20)
    now = datetime.datetime.now()
    print
    now.strftime('%Y-%m-%d %H:%M:%S')
    print('get n-grams')
    df_features['f_1dis'] = pool.map(get_word_distance, df_features['questions'])
    df_features['f_2word_dis'] = pool.map(get_word_2gram_distance, df_features['questions'])
    df_features['f_2char_dis'] = pool.map(get_char_2gram_distance, df_features['questions'])
    df_features['f_3word_dis'] = pool.map(get_word_3gram_distance, df_features['questions'])
    df_features['f_3char_dis'] = pool.map(get_char_3gram_distance, df_features['questions'])

    df_features['f_1dis2'] = pool.map(get_word_distance2, df_features['questions'])
    df_features['f_2word_dis2'] = pool.map(get_word_2gram_distance2, df_features['questions'])
    df_features['f_2char_dis2'] = pool.map(get_char_2gram_distance2, df_features['questions'])
    df_features['f_3word_dis2'] = pool.map(get_word_3gram_distance2, df_features['questions'])
    df_features['f_3char_dis2'] = pool.map(get_char_3gram_distance2, df_features['questions'])

    df_features['f_1dis3'] = pool.map(get_word_distance3, df_features['questions'])
    df_features['f_2word_dis3'] = pool.map(get_word_2gram_distance3, df_features['questions'])
    df_features['f_2char_dis3'] = pool.map(get_char_2gram_distance3, df_features['questions'])
    df_features['f_3word_dis3'] = pool.map(get_word_3gram_distance3, df_features['questions'])
    df_features['f_3char_dis3'] = pool.map(get_char_3gram_distance3, df_features['questions'])

    df_features['f_1dis4'] = pool.map(get_word_distance4, df_features['questions'])
    df_features['f_2word_dis4'] = pool.map(get_word_2gram_distance4, df_features['questions'])
    df_features['f_2char_dis4'] = pool.map(get_char_2gram_distance4, df_features['questions'])
    df_features['f_3word_dis4'] = pool.map(get_word_3gram_distance4, df_features['questions'])
    df_features['f_3char_dis4'] = pool.map(get_char_3gram_distance4, df_features['questions'])

    print('all done')
    now = datetime.datetime.now()
    print
    now.strftime('%Y-%m-%d %H:%M:%S')
    df_features.fillna(0.0)
    return df_features


if __name__ == "__main__":
    train['questions'] = train['question1'] + '_split_tag_' + train['question2']
    test['questions'] = test['question1'] + '_split_tag_' + test['question2']

    train = makeFeature(train)
    train.to_csv('train_gram_feature.csv', index=False)

    test = makeFeature(test)
    test.to_csv('test_gram_feature.csv', index=False)
