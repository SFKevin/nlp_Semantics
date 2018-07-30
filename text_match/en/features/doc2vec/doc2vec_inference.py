import numpy as np
import pandas as pd
import gensim
from text_match.en.data_utils import datahelper
import datetime
from text_match.en.features.doc2vec import doc2vec_model

# filepath_en_train = "E:\\CIKM2018\\cikm_english_train_20180516\\cikm_english_train_20180516.txt"
# filepath_sp_train = "E:\\CIKM2018\\cikm_spanish_train_20180516.txt"
# filepath_test = "E:\\CIKM2018\\cikm_test_a_20180516.txt"
# filepath_unlabel = "E:\\CIKM2018\\cikm_unlabel_spanish_train_20180516\\cikm_unlabel_spanish_train_20180516.txt"
# w2v_pah = "E:\\CIKM2018\\w2v.model.bin"
# fast_path = "E:\\CIKM2018\\fast_text_vectors_wiki.es.vec\\wiki.es.vec"
# file_stop_word = "E:\\CIKM2018\\spanish_stop_word.txt"
#
#
filepath_en_train = "I:\\CIKM\\cikm_english_train_20180516\\cikm_english_train_20180516.txt"
filepath_sp_train = "I:\\CIKM\\cikm_spanish_train_20180516.txt"
filepath_test = "I:\\CIKM\\cikm_test_a_20180516.txt"
filepath_unlabel = "I:\\CIKM\\cikm_unlabel_spanish_train_20180516\\cikm_unlabel_spanish_train_20180516.txt"
w2v_pah = "I:\\CIKM\\w2v.model.bin"
fast_path = "I:\\CIKM\\fast_text_vectors_wiki.es.vec\\wiki.es.vec"
file_stop_word = "I:\\CIKM\\spanish_stop_word.txt"


def Cosine(vec1, vec2):
    vec1 = np.array(vec1, dtype=np.float)
    vec2 = np.array(vec2, dtype=np.float)
    Lx = np.sqrt(vec1.dot(vec1))
    Ly = np.sqrt(vec2.dot(vec2))
    return vec1.dot(vec2) / ((Lx * Ly) + 0.000001)


def Manhatton(vec1, vec2):
    return np.sum(np.fabs(np.array(vec1, dtype=np.float) - np.array(vec2, dtype=np.float)))


def Euclidean(vec1, vec2):
    return np.sqrt(np.sum(np.array(vec1, dtype=np.float) - np.array(vec2, dtype=np.float)) ** 2)


def PearsonSimilar(vec1, vec2):
    vec1 = np.array(vec1, dtype=np.float)
    vec2 = np.array(vec2, dtype=np.float)
    data = np.vstack((vec1, vec2))
    return pd.DataFrame(data).T.corr('pearson')[0][1]


def SpearmanSimilar(vec1, vec2):
    vec1 = np.array(vec1, dtype=np.float)
    vec2 = np.array(vec2, dtype=np.float)
    data = np.vstack((vec1, vec2))
    return pd.DataFrame(data).T.corr('spearman')[0][1]


def KendallSimilar(vec1, vec2):
    vec1 = np.array(vec1, dtype=np.float)
    vec2 = np.array(vec2, dtype=np.float)
    data = np.vstack((vec1, vec2))
    return pd.DataFrame(data).T.corr('kendall')[0][1]


model_name = "%s" % ("doc_2_vec_1532938738")
model_saved_file = "%s" % (model_name)
model = gensim.models.doc2vec.Doc2Vec.load(model_saved_file)


def process_data(inputs_data):
    d2c_list = []
    for line in inputs_data:
        # line_list = [x for x in line if x not in stop_word_list]
        line = datahelper.text_to_wordlist(line, remove_stop_words=True, stem_words=False).split(" ")
        d2c_list.append(line)
    return d2c_list


def makeFeature():
    x_train1, x_train2, _ = datahelper.load_data(filepath_en_train, filepath_sp_train)

    x_train1 = process_data(x_train1)
    x_train2 = process_data(x_train2)

    now = datetime.datetime.now()
    print
    now.strftime('%Y-%m-%d %H:%M:%S')
    print('get sentence vector')
    train = pd.DataFrame()
    train['doc2vec_train1'] = [doc2vec_model.get_question_vector(x, model) for x in x_train1]
    train['doc2vec_train2'] = [doc2vec_model.get_question_vector(x, model) for x in x_train2]
    print('get six kinds of coefficient about vector')

    train['cosine1'] = train.apply(lambda x: Cosine(x['doc2vec_train1'], x['doc2vec_train2']), axis=1)
    train['manhatton1'] = train.apply(lambda x: Manhatton(x['doc2vec_train1'], x['doc2vec_train2']), axis=1)
    train['euclidean1'] = train.apply(lambda x: Euclidean(x['doc2vec_train1'], x['doc2vec_train2']), axis=1)
    train['pearson1'] = train.apply(lambda x: PearsonSimilar(x['doc2vec_train1'], x['doc2vec_train2']), axis=1)
    train['spearman1'] = train.apply(lambda x: SpearmanSimilar(x['doc2vec_train1'], x['doc2vec_train2']),
                                     axis=1)
    train['kendall1'] = train.apply(lambda x: KendallSimilar(x['doc2vec_train1'], x['doc2vec_train2']),
                                    axis=1)
    train.to_csv('train_doc2vec1.csv', index=False)


if __name__ == '__main__':
    makeFeature()
