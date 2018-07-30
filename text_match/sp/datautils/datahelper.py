# filepath_en_train = "E:\\CIKM2018\\cikm_english_train_20180516\\cikm_english_train_20180516.txt"
# filepath_sp_train = "E:\\CIKM2018\\cikm_spanish_train_20180516.txt"
# filepath_test = "E:\\CIKM2018\\cikm_test_a_20180516.txt"
# filepath_unlabel = "E:\\CIKM2018\\cikm_unlabel_spanish_train_20180516\\cikm_unlabel_spanish_train_20180516.txt"
# w2v_pah = "E:\\CIKM2018\\w2v.model.bin"
# fast_path = "E:\\CIKM2018\\fast_text_vectors_wiki.es.vec\\wiki.es.vec"
filepath_en_train = "I:\\CIKM\\cikm_english_train_20180516\\cikm_english_train_20180516.txt"
filepath_sp_train = "I:\\CIKM\\cikm_spanish_train_20180516.txt"
filepath_test = "I:\\CIKM\\cikm_test_a_20180516.txt"
filepath_unlabel = "I:\\CIKM\\cikm_unlabel_spanish_train_20180516\\cikm_unlabel_spanish_train_20180516.txt"
w2v_pah = "I:\\CIKM\\w2v.model.bin"
fast_path = "I:\\CIKM\\fast_text_vectors_wiki.es.vec\\wiki.es.vec"
file_stop_word = "I:\\CIKM\\spanish_stop_word.txt"
import numpy as np
from gensim.models import KeyedVectors
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
from string import punctuation

# from nltk.corpus import stopwords
#
# stops = set(stopwords.words("spanish"))
# stop_words = nltk
PAD_ID = 0
UNK_ID = 1
_PAD = "_PAD"
_UNK = "UNK"


def load_data(en_train, sp_train):
    en_train = list(open(en_train, "r", encoding='UTF-8').readlines())
    sp_train = list(open(sp_train, "r", encoding='UTF-8').readlines())
    unlabel_data = list(open(filepath_unlabel, "r", encoding='UTF-8').readlines())
    test_data = list(open(filepath_test, "r", encoding='UTF-8').readlines())

    en_train_list = [
        line.strip().replace("\n", " ").replace(",", " ").replace(".", " ").replace("?", " ").replace("¿", " ").replace(
            "!",
            " ").replace(
            "¡", " ").replace("_", " ").replace("-", " ").replace("/", " ").replace("\\", " ").replace("\'",
                                                                                                       " ").lower().split(
            "\t") for line
        in
        en_train]
    sp_train_list = [
        line.strip().replace("\n", " ").replace(",", " ").replace(".", " ").replace("?", " ").replace("¿", " ").replace(
            "!",
            " ").replace(
            "¡", " ").replace("_", " ").replace("-", " ").replace("/", " ").replace("\\", " ").replace("\'",
                                                                                                       " ").lower().split(
            "\t") for line
        in
        sp_train]
    unlabel_data_list = [
        line.strip().replace("\n", " ").replace(",", " ").replace(".", " ").replace("?", " ").replace("¿", " ").replace(
            "!",
            " ").replace(
            "¡", " ").split(
            "\t")[0].replace("_", " ").replace("-", " ").replace("/", " ").replace("\\", " ").replace("\'", " ").lower()
        for
        line in
        unlabel_data]

    test_data_list = [
        line.strip().replace("\n", " ").replace(",", " ").replace(".", " ").replace("?", " ").replace("¿", " ").replace(
            "!",
            " ").replace(
            "¡", " ").replace("_", " ").replace("-", " ").replace("/", " ").replace("\\", " ").replace("\'",
                                                                                                       " ").lower().split(
            "\t") for
        line in
        test_data]

    sp_pair_en = [[x[1].lower().strip(), x[3].lower().strip()] for x in en_train_list]
    sp_score_en = [x[4].lower().strip() for x in en_train_list]

    sp_pair_sp = [[x[0].lower().strip(), x[2].lower().strip()] for x in sp_train_list]
    sp_score_sp = [x[4].lower().strip() for x in sp_train_list]

    x_train = np.concatenate([sp_pair_en, sp_pair_sp], axis=0)
    y_train = sp_score_en + sp_score_sp
    y_train = [int(y) for y in y_train]
    y_train = np.array(y_train)

    x_train_reshape = list(x_train.reshape(-1))
    test_reshape = list(np.array(test_data_list).reshape(-1))

    x_train1 = [x[0].lower().strip() for x in x_train]
    x_train2 = [x[1].lower().strip() for x in x_train]
    w2v = x_train_reshape + unlabel_data_list + test_reshape
    w2v_list = [word_tokenize(x) for x in w2v]
    del en_train, sp_train, unlabel_data, test_data, en_train_list, sp_train_list, unlabel_data_list, test_data_list
    del sp_pair_en, sp_score_en, sp_pair_sp, sp_score_sp, test_reshape
    return x_train1, x_train2, x_train, y_train, w2v_list, w2v


def load_testdata(test_file):
    test_data = list(open(test_file, "r", encoding='UTF-8').readlines())
    test_data_list = [
        line.replace("\n", " ").replace(",", " ").replace(".", " ").replace("?", " ").replace("¿", " ").replace("!",
                                                                                                                "").replace(
            "¡", " ").replace("_", " ").replace("-", " ").replace("/", " ").replace("\\", " ").replace("\'",
                                                                                                       " ").lower().split(
            "\t") for
        line in
        test_data]
    test_data1 = [x[0] for x in test_data_list]
    test_data2 = [x[1] for x in test_data_list]
    return test_data1, test_data2


def create_vocabulary(x_text):
    index2word = {}
    index2word[PAD_ID] = _PAD
    index2word[UNK_ID] = _UNK

    word2index = {}
    word2index[_PAD] = PAD_ID
    word2index[_UNK] = UNK_ID

    c_inputs = Counter()
    for text in x_text:
        text = word_tokenize(text)
        # text = [c for c in text if c not in stops]
        text = " ".join([c for c in text if c not in punctuation])

        text = text.split()
        # stemmer = SnowballStemmer('spanish')
        # stemmed_words = [stemmer.stem(word.lower()) for word in text]
        c_inputs.update(text)
    vocab_list = c_inputs.most_common(100000)
    for i, tuplee in enumerate(vocab_list):
        word, _ = tuplee
        word2index[word] = i + 2
        index2word[i + 2] = word
    return word2index, index2word


def asign_pretrained_word_embedding(vocabulary_index2word, vocab_size, word2vec_model_path):
    # word2vec_model = fasttext.load_model(fast_path)
    # word2vec_model = KeyedVectors.load_word2vec_format(fast_path)
    word2vec_model = KeyedVectors.load_word2vec_format(w2v_pah, binary=True)
    word2vec_dict = {}
    # not_exist = []
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)
    # word_embedding_2dlist[0] = np.random.uniform(-bound, bound, 300)
    word_embedding_2dlist[0] = np.zeros(300)
    # count_exist = 0
    # count_not_exist = 0
    np.random.seed(1000)
    for i in range(1, vocab_size):
        word = vocabulary_index2word[i]
        # embedding = None
        try:
            embedding = word2vec_dict[word]
        except Exception:
            embedding = None
        if embedding is not None:
            word_embedding_2dlist[i] = embedding
            # count_exist = count_exist + 1
        else:
            # not_exist.append(word)
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, 300)
            # count_not_exist = count_not_exist + 1
    word_embedding_final = np.array(word_embedding_2dlist)
    # print("not exist: %d" % count_not_exist)
    return word_embedding_final


def asign_pretrained_word_embedding_cnn(vocabulary_index2word, vocab_size, word2vec_model_path):
    # word2vec_model = fasttext.load_model(fast_path)
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)
    # word_embedding_2dlist[0] = np.random.uniform(-bound, bound, 300)
    word_embedding_2dlist[0] = np.zeros(300)
    count_exist = 0
    count_not_exist = 0
    np.random.seed(1000)
    for i in range(1, vocab_size):
        word = vocabulary_index2word[i]
        try:
            embedding = word2vec_dict[word]
        except Exception:
            embedding = None
        if embedding is not None:
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1
        else:
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, 300)
            # count_not_exist = count_not_exist + 1
    word_embedding_final = np.array(word_embedding_2dlist)
    # print("not exist: %d" % count_not_exist)
    return word_embedding_final


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    _, _, _, _, _, x_text = load_data(filepath_en_train, filepath_sp_train)
    # count0 = 0
    # count1 = 0
    # total = 0
    # for i in range(len(y_train)):
    #     total = total + 1
    #     if y_train[i] == 1:
    #         count1 = count1 + 1
    #     elif y_train[i] == 0:
    #         count0 = count0 + 1
    # print("total number: %d, 1 number: %d, 0 number: %d" % (total, count1, count0))
    wo2in, in2wo = create_vocabulary(x_text)
    voca_size = len(in2wo)
    embed = asign_pretrained_word_embedding(in2wo, voca_size, w2v_pah)
    # for i in range(len(not_exist)):
    #     print(not_exist[i])
    print("voca_size: %d" % voca_size)
    print("embedding size: %d" % len(embed))
