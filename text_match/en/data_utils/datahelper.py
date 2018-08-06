filepath_en_train = "E:\\CIKM2018\\cikm_english_train_20180516\\cikm_english_train_20180516.txt"
filepath_sp_train = "E:\\CIKM2018\\cikm_spanish_train_20180516.txt"
filepath_test = "E:\\CIKM2018\\cikm_test_a_20180516.txt"
filepath_unlabel = "E:\\CIKM2018\\cikm_unlabel_spanish_train_20180516\\cikm_unlabel_spanish_train_20180516.txt"
# filepath_en_train = "I:\\CIKM\\cikm_english_train_20180516\\cikm_english_train_20180516.txt"
# filepath_sp_train = "I:\\CIKM\\cikm_spanish_train_20180516.txt"
# filepath_test = "I:\\CIKM\\cikm_test_a_20180516.txt"
# filepath_unlabel = "I:\\CIKM\\cikm_unlabel_spanish_train_20180516\\cikm_unlabel_spanish_train_20180516.txt"
# w2v_pah = "I:\\CIKM\\GoogleNews-vectors-negative300.bin"
# fast_path = "I:\\CIKM\\fast_text_vectors_wiki.en.vec\\wiki.en.vec"
# file_stop_word = "I:\\CIKM\\spanish_stop_word.txt"

import numpy as np
import re
from string import punctuation
from nltk.corpus import stopwords
import pandas as pd

stop_words = set(stopwords.words("english"))
import nltk


def load_data(en_train, sp_train):
    en_train = list(open(en_train, "r", encoding='UTF-8').readlines())
    sp_train = list(open(sp_train, "r", encoding='UTF-8').readlines())

    en_train_list = [line.strip().replace("\n", "").split("\t") for line in en_train]
    sp_train_list = [line.strip().replace("\n", "").split("\t") for line in sp_train]

    sp_pair_en = [[x[0].lower().strip(), x[2].lower().strip()] for x in en_train_list]
    sp_score_en = [x[4].lower().strip() for x in en_train_list]

    sp_pair_sp = [[x[1].lower().strip(), x[2].lower().strip()] for x in sp_train_list]
    sp_score_sp = [x[4].lower().strip() for x in sp_train_list]
    x_train = np.concatenate([sp_pair_en, sp_pair_sp], axis=0)
    y_train = sp_score_en + sp_score_sp
    y_train = [int(y) for y in y_train]
    y_train = np.array(y_train)

    x_train1 = [x[0].strip() for x in x_train]
    x_train2 = [x[1].strip() for x in x_train]

    return x_train1, x_train2, y_train


def load_data_over(en_train, sp_train):
    en_train = list(open(en_train, "r", encoding='UTF-8').readlines())
    sp_train = list(open(sp_train, "r", encoding='UTF-8').readlines())

    en_train_list = [line.strip().replace("\n", "").split("\t") for line in en_train]
    sp_train_list = [line.strip().replace("\n", "").split("\t") for line in sp_train]

    sp_pair_en = [[x[0].lower().strip(), x[2].lower().strip()] for x in en_train_list]
    sp_score_en = [x[4].lower().strip() for x in en_train_list]

    sp_pair_sp = [[x[1].lower().strip(), x[2].lower().strip()] for x in sp_train_list]
    sp_score_sp = [x[4].lower().strip() for x in sp_train_list]
    x_train = np.concatenate([sp_pair_en, sp_pair_sp], axis=0)
    y_train = sp_score_en + sp_score_sp
    y_train = [int(y) for y in y_train]
    y_train = np.array(y_train)

    x_train1 = [x[0].strip() for x in x_train]
    x_train2 = [x[1].strip() for x in x_train]
    data = pd.DataFrame()
    data['x1'] = x_train1
    data['x2'] = x_train2
    data['y'] = y_train
    y1 = data[data['y'] == 1]
    y1_sample = y1.sample(frac=0.05, axis=0)
    data_over = pd.concat([data, y1_sample], axis=0)
    x_train1 = data_over['x1'].values
    x_train2 = data_over['x2'].values
    y_train = data_over['y'].values
    return x_train1, x_train2, y_train


def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    text = text.rstrip('?')
    text = text.rstrip(',')
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text)
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text)
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)

    text = ''.join([c for c in text if c not in punctuation])

    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    if stem_words:
        text = text.split()
        stemmed_words = [nltk.PorterStemmer().stem(word.lower()) for word in text]
        text = " ".join(stemmed_words)
    return text


PAD_ID = 0
UNK_ID = 1
_PAD = "_PAD"
_UNK = "UNK"
from collections import Counter
from nltk.tokenize import word_tokenize


def create_vocabulary(x_text):
    index2word = {}
    index2word[PAD_ID] = _PAD
    index2word[UNK_ID] = _UNK

    word2index = {}
    word2index[_PAD] = PAD_ID
    word2index[_UNK] = UNK_ID

    c_inputs = Counter()
    for text in x_text:
        text = text_to_wordlist(text, remove_stop_words=True, stem_words=False)
        text = word_tokenize(text)
        c_inputs.update(text)
    vocab_list = c_inputs.most_common(1000000)
    for i, tuplee in enumerate(vocab_list):
        word, _ = tuplee
        word2index[word] = i + 2
        index2word[i + 2] = word
    return word2index, index2word


import pickle


def asign_pretrained_word_embedding():
    # word2vec_model = KeyedVectors.load_word2vec_format(w2v_pah, binary=True)
    # word2vec_model = KeyedVectors.load_word2vec_format(fast_path)
    # word2vec_dict = {}
    # not_exist = []
    # for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
    #     word2vec_dict[word] = vector
    # word_embedding_2dlist = [[]] * vocab_size
    # bound = np.sqrt(6.0) / np.sqrt(vocab_size)
    # word_embedding_2dlist[0] = np.random.uniform(-bound, bound, 300)
    # word_embedding_2dlist[0] = np.zeros(300)
    # count_exist = 0
    # count_not_exist = 0
    # np.random.seed(1000)
    # for i in range(1, vocab_size):
    #     word = vocabulary_index2word[i]
    embedding = None
    # try:
    #     embedding = word2vec_dict[word]
    # except Exception:
    #     embedding = None
    # if embedding is not None:
    #     word_embedding_2dlist[i] = embedding
    #     count_exist = count_exist + 1
    # else:
    #     not_exist.append(word)
    #     word_embedding_2dlist[i] = np.random.uniform(-bound, bound, 300)
    #     count_not_exist = count_not_exist + 1
    # word_embedding_final = np.array(word_embedding_2dlist)
    # print("not exist: %d" % count_not_exist)
    # return word_embedding_final, not_exist
    with open('I:\\nlp_semantics\\text_match\\en\\data_utils\\embed.pickle', 'rb') as f:
        embed = pickle.load(f)
    return np.array(embed)


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
    x_train1, x_train2, _ = load_data(filepath_en_train, filepath_sp_train)
    x_text = np.concatenate([x_train1, x_train2], axis=0)
    wo2in, in2wo = create_vocabulary(x_text)
    voca_size = len(in2wo)
    # with open('embed.pickle', 'rb') as f:
    #     embed = pickle.load(f)
    embed = asign_pretrained_word_embedding(in2wo, voca_size)
    with open('embed.pickle', 'wb') as f:
        pickle.dump(embed, f, protocol=pickle.HIGHEST_PROTOCOL)
    # for i in range(len(not_exist)):
    #     print(not_exist[i])
    # print("voca_size: %d" % voca_size)
    print("embedding size: %d" % len(embed))
    print(embed[0])
