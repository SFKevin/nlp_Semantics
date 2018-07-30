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
from gensim.models.tfidfmodel import TfidfModel
from gensim.similarities import MatrixSimilarity
from scipy import spatial
import datetime
from scipy.stats import skew, kurtosis
from gensim.corpora.dictionary import Dictionary

x_train1, x_train2, _, _, _, _ = datahelper.load_data(filepath_en_train, filepath_sp_train)
x_test1, x_test2 = datahelper.load_testdata(filepath_test)
train = pd.DataFrame()
test = pd.DataFrame()

train['question1'] = x_train1
train['question2'] = x_train2

test['question1'] = x_test1
test['question2'] = x_test2

# clean
tfidf_txt = train['question1'].tolist() + train['question2'].tolist() + test['question1'].tolist() + test[
    'question2'].tolist()
train_qs = pd.Series(tfidf_txt).astype(str)
dictionary = Dictionary(x.split(" ") for x in tfidf_txt)


class MyCorpus(object):
    def __iter__(self):
        for x in tfidf_txt:
            yield dictionary.doc2bow(x.split(" "))


corpus = MyCorpus()
tfidf = TfidfModel(corpus)


def getdiffwords(q1, q2):
    word1 = q1.split()
    word2 = q2.split()
    qdf1 = [w for w in word1 if w not in word2]
    tmp = " ".join(qdf1)
    return tmp


def to_tfidf(text):
    res = tfidf[dictionary.doc2bow(text.split(" "))]
    return res


def tfidf_w(token):
    weights = dictionary.token2id
    if token in weights:
        res = tfidf.idfs[weights[token]]
    else:
        res = 1.0
    return res


def eucldist_vectorized(word_1, word_2):
    try:
        w2v1 = model[word_1]
        w2v2 = model[word_2]
        sim = np.sqrt(np.sum((np.array(w2v1) - np.array(w2v2)) ** 2))
        return float(sim)
    except:
        return float(0)


# 输入两个wordlist
# 默认句子中每个词权重相同，实际可以更改
def getDiff(wordlist_1, wordlist_2):
    wordlist_1 = wordlist_1.split()
    wordlist_2 = wordlist_2.split()
    num = len(wordlist_1) + 0.001
    sim = 0.0
    for word_1 in wordlist_1:
        dis = 0.0
        for word_2 in wordlist_2:
            if (dis == 0.0):
                dis = eucldist_vectorized(word_1, word_2)
            else:
                dis = min(dis, eucldist_vectorized(word_1, word_2))
        sim += dis
    return (sim / num)


def getDiff_weight(wordlist_1, wordlist_2):
    wordlist_1 = wordlist_1.split()
    wordlist_2 = wordlist_2.split()
    tot_weights = 0.0
    for word_1 in wordlist_1:
        tot_weights += tfidf_w(word_1)
    sim = 0.0
    for word_1 in wordlist_1:
        dis = 0.0
        for word_2 in wordlist_2:
            if (dis == 0.0):
                dis = eucldist_vectorized(word_1, word_2)
            else:
                dis = min(dis, eucldist_vectorized(word_1, word_2))
        sim += tfidf_w(word_1) * dis
    return sim


def getDiff_averge(wordlist_1, wordlist_2):
    return getDiff_weight(wordlist_1, wordlist_2) + getDiff_weight(wordlist_2, wordlist_1)


def cos_sim(text1, text2):
    tfidf1 = to_tfidf(text1)
    tfidf2 = to_tfidf(text2)
    index = MatrixSimilarity([tfidf1], num_features=len(dictionary))
    sim = index[tfidf2]
    return float(sim[0])


def get_vector(text):
    # 建立一个全是0的array
    res = np.zeros([300])
    count = 0
    for word in text.split(" "):
        if word in vocab:
            res += tfidf_w(word) * model[word]
            count += tfidf_w(word)
    if count != 0:
        return res / count
    return np.zeros([300])


def get_weight_vector(text):
    # 建立一个全是0的array
    res = np.zeros([300])
    count = 0
    for word in text.split(" "):
        if word in vocab:
            res += model[word]
            count += 1
    if count != 0:
        return res / count
    return np.zeros([300])


def w2v_cos_sim(text1, text2):
    try:
        w2v1 = get_weight_vector(text1)
        w2v2 = get_weight_vector(text2)
        sim = 1 - spatial.distance.cosine(w2v1, w2v2)
        return float(sim)
    except:
        return float(0)


def get_features(df_features):
    print('use w2v to document presentation')
    now = datetime.datetime.now()
    now.strftime('%Y-%m-%d %H:%M:%S')
    print('get_w2v')
    now = datetime.datetime.now()
    now.strftime('%Y-%m-%d %H:%M:%S')
    df_features['q1_unique'] = df_features.apply(lambda x: getdiffwords(x['question1'], x['question2']), axis=1)
    df_features['q2_unique'] = df_features.apply(lambda x: getdiffwords(x['question2'], x['question1']), axis=1)

    df_features['q1_unique_w2v_weight'] = df_features.q1_unique.map(lambda x: get_vector(x))
    df_features['q2_unique_w2v_weight'] = df_features.q2_unique.map(lambda x: get_vector(x))
    df_features['q1_unique_w2v'] = df_features.q1_unique.map(lambda x: get_weight_vector(x))
    df_features['q2_unique_w2v'] = df_features.q2_unique.map(lambda x: get_weight_vector(x))

    print('z_dist')
    now = datetime.datetime.now()
    now.strftime('%Y-%m-%d %H:%M:%S')
    now = datetime.datetime.now()
    print('z_tfidf_cos_sim')
    now.strftime('%Y-%m-%d %H:%M:%S')
    now = datetime.datetime.now()
    now.strftime('%Y-%m-%d %H:%M:%S')

    df_features['z_w2v_unique_dis_e_weight'] = df_features.apply(
        lambda x: spatial.distance.euclidean(x['q1_unique_w2v_weight'], x['q2_unique_w2v_weight']), axis=1)
    df_features['z_w2v_unique_dis_e'] = df_features.apply(
        lambda x: spatial.distance.euclidean(x['q1_unique_w2v'], x['q2_unique_w2v']), axis=1)

    df_features['z_w2v_unique_dis_mink_w'] = df_features.apply(
        lambda x: spatial.distance.minkowski(x['q1_unique_w2v_weight'], x['q2_unique_w2v_weight'], 3), axis=1)
    df_features['z_w2v_unique_dis_cityblock_w'] = df_features.apply(
        lambda x: spatial.distance.cityblock(x['q1_unique_w2v_weight'], x['q2_unique_w2v_weight']), axis=1)
    df_features['z_w2v_unique_dis_canberra_w'] = df_features.apply(
        lambda x: spatial.distance.canberra(x['q1_unique_w2v_weight'], x['q2_unique_w2v_weight']), axis=1)

    df_features['z_w2v_unique_dis_mink'] = df_features.apply(
        lambda x: spatial.distance.minkowski(x['q1_unique_w2v'], x['q2_unique_w2v'], 3), axis=1)
    df_features['z_w2v_unique_dis_cityblock'] = df_features.apply(
        lambda x: spatial.distance.cityblock(x['q1_unique_w2v'], x['q2_unique_w2v']), axis=1)
    df_features['z_w2v_unique_dis_canberra'] = df_features.apply(
        lambda x: spatial.distance.canberra(x['q1_unique_w2v'], x['q2_unique_w2v']), axis=1)

    df_features['z_q1_unique_skew_w'] = df_features.q1_unique_w2v_weight.map(lambda x: skew(x))
    df_features['z_q2_unique_skew_w'] = df_features.q2_unique_w2v_weight.map(lambda x: skew(x))
    df_features['z_q1_unique_kur_w'] = df_features.q1_unique_w2v_weight.map(lambda x: kurtosis(x))
    df_features['z_q2_unique_kur_w'] = df_features.q2_unique_w2v_weight.map(lambda x: kurtosis(x))

    df_features['z_q1_unique_skew'] = df_features.q1_unique_w2v.map(lambda x: skew(x))
    df_features['z_q2_unique_skew'] = df_features.q2_unique_w2v.map(lambda x: skew(x))
    df_features['z_q1_unique_kur'] = df_features.q1_unique_w2v.map(lambda x: kurtosis(x))
    df_features['z_q2_unique_kur'] = df_features.q2_unique_w2v.map(lambda x: kurtosis(x))
    del df_features['q1_unique_w2v_weight']
    del df_features['q2_unique_w2v_weight']
    del df_features['q1_unique_w2v']
    del df_features['q2_unique_w2v']
    print('all done')
    now.strftime('%Y-%m-%d %H:%M:%S')
    df_features.fillna(0.0)
    return df_features


from gensim.models import KeyedVectors

if __name__ == '__main__':
    model = KeyedVectors.load_word2vec_format(w2v_pah, binary=True)
    vocab = model.vocab

    train = get_features(train)
    train.to_csv('train_weight_noweight.csv', index=False)

    test = get_features(test)
    test.to_csv('test_weight_noweight.csv', index=False)
