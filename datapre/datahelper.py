import re
import numpy as np
from collections import Counter
# //import tensorflow as tf
from gensim.models import KeyedVectors
# from tflearn.data_utils import pad_sequences
import random
PAD_ID = 0
UNK_ID = 1
_PAD = "_PAD"
_UNK = "UNK"


# tf.flags.DEFINE_integer("vocab_size", 100000, "maximum vocab size.")
# # tf.flags.DEFINE_string("positive_data_file", "../rt-polaritydata/rt-polarity.pos", "positive_data_file not found")
# # tf.flags.DEFINE_string("negative_data_file", "../rt-polaritydata/rt-polarity.neg", "negative_data_file not found")
# # tf.flags.DEFINE_string("word2vec_model_path", "../GoogleNews-vectors-negative300.bin", "Goole News Vector")
# # tf.flags.DEFINE_integer("embed_size", 300, "embedding size")
# FLAGS = tf.flags.FLAGS


def asign_pretrained_word_embedding(vocabulary_index2word, vocab_size, word2vec_model_path):
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    # print(word2vec_model['boy'])
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)
    word_embedding_2dlist[0] = np.random.uniform(-bound, bound, 300)
    count_exist = 0
    count_not_exist = 0
    for i in range(1, vocab_size):
        word = vocabulary_index2word[i]
        embedding = None
        try:
            embedding = word2vec_dict[word]
        except Exception:
            embedding = None
        if embedding is not None:
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1
        else:
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, 300)
            count_not_exist = count_not_exist + 1
    word_embedding_final = np.array(word_embedding_2dlist)
    # print("finished!\n")
    # word_embedding = tf.constant(word_embedding_final, tf.float32)
    return word_embedding_final


def clean_str(string):
    """
    :param string:
    :return:
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def create_vocabulary(x_text):
    vocabulary_word2index = {}
    vocabulary_index2word = {}
    vocabulary_word2index[_PAD] = PAD_ID
    vocabulary_index2word[PAD_ID] = _PAD
    vocabulary_word2index[_UNK] = UNK_ID
    vocabulary_index2word[UNK_ID] = _UNK

    c_inputs = Counter()
    for text in x_text:
        text = text.split(" ")
        c_inputs.update(text)
    vocab_list = c_inputs.most_common(100000)

    for i, tuplee in enumerate(vocab_list):
        word, _ = tuplee
        vocabulary_word2index[word] = i + 2
        vocabulary_index2word[i + 2] = word
    return vocabulary_word2index, vocabulary_index2word


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Load MR polarity data from files, splits the data into words and generate labels.
    :param positive_data_file:
    :param negative_data_file:
    :return: sentence and labels
    """
    positive_examples = list(
        open(
            positive_data_file,
            "r",
            encoding='UTF-8').readlines())
    negative_examples = list(
        open(
            negative_data_file,
            "r",
            encoding='UTF-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]
    x_text = positive_examples + negative_examples
    # x_text = [clean_str(sent) for sent in x_text]
    # labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_lables = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_lables], 0)
    return [x_text, y]


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


def convert2fasttext(positive_data_file, negative_data_file):
    """
    convert the data to fasttext format data
    :param positive_data_file:
    :param negative_data_file:
    :return:
    """
    positive_examples = list(
        open(
            positive_data_file,
            "r",
            encoding='UTF-8').readlines())
    negative_examples = list(
        open(
            negative_data_file,
            "r",
            encoding='UTF-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x = [text.split(" ") for text in x_text]
    positive_labels = ["__label__pos" for _ in positive_examples]
    negative_lables = ["__label__neg" for _ in negative_examples]
    y = positive_labels + negative_lables
    data_ziped = list(zip(x, y))
    random.shuffle(data_ziped)
    dev_sample_index = -1 * int(0.1 * float(len(data_ziped)))
    train = data_ziped[:dev_sample_index]
    test = data_ziped[dev_sample_index:]
    fp_train = open("../rt-polaritydata/train.txt", "w")
    for line in train:
        i = 0
        for words in line:
            for word in words:
                fp_train.write(word)
                if i == 0:
                    fp_train.write(" ")
            i = i + 1
        fp_train.write("\n")
    fp_train.close()
    fp_test = open("../rt-polaritydata/test.txt", "w")
    for line in test:
        i = 0
        for words in line:
            for word in words:
                fp_test.write(word)
                if i == 0:
                    fp_test.write(" ")
            i = i + 1
        fp_test.write("\n")
    fp_test.close()


###################################################################################################
if __name__ == '__main__':
    train_path = "../rt-polaritydata/train.txt"
    test_path = "../rt-polaritydata/test.txt"
    classifier = fasttext.supervised(train_path, '../rt-polaritydata/fasttext.model')
    # convert2fasttext(positive_file, negative_file)
    # x_text, y = load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    # vocabulary_word2index, vocabulary_index2word = create_vocabulary(x_text)
    # x_int_text = []
    # for line in x_text:
    #     input_list = line.split(" ")
    #     text = [vocabulary_word2index.get(x, UNK_ID) for x in input_list]
    #     x_int_text.append(text)
    # print("vocabulary size: %d" % len(vocabulary_word2index))
    # vocab_size = len(vocabulary_word2index)
    # xtext = pad_sequences(x_int_text, maxlen=56, value=0.0)
    # embedding = asign_pretrained_word_embedding(vocabulary_index2word, vocab_size, FLAGS.word2vec_model_path)
