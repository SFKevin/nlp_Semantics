import re
import numpy as np

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
alphabet_size = len(alphabet)
l0 = 1014


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


def data_reader(positive_data_file, negative_data_file):
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
    embedding_w, embedding_dic = onehot_dic_build()
    x_image = []
    maxlen = np.max([len(x) for x in x_text])
    for i in range(len(x_text)):
        doc_vec = doc_process(maxlen, x_text[i], embedding_dic)
        x_image.append(doc_vec)
    return x_image, y,maxlen


def doc_process(maxlen, text, embedding_dic):
    textlen = len(text)
    doc_vec = np.zeros(maxlen, dtype='int32')
    for j in range(textlen):
        if text[j] in embedding_dic:
            doc_vec[j] = embedding_dic[text[j]]
        else:
            doc_vec[j] = embedding_dic['UNK']
    for j in range(textlen, maxlen):
        doc_vec[j] = embedding_dic['UNK']
    return doc_vec


def onehot_dic_build():
    # onehot编码
    embedding_dic = {}
    embedding_w = []
    # 对于字母表中不存在的或者空的字符用全0向量代替
    embedding_dic["UNK"] = 0
    embedding_w.append(np.zeros(len(alphabet), dtype='float32'))

    for i, alpha in enumerate(alphabet):
        onehot = np.zeros(len(alphabet), dtype='float32')
        embedding_dic[alpha] = i + 1
        onehot[i] = 1
        embedding_w.append(onehot)

    embedding_w = np.array(embedding_w, dtype='float32')
    return embedding_w, embedding_dic

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

pos_path = "E:\\data\\rt-polaritydata\\rt-polarity.pos"
neg_path = "E:\\data\\rt-polaritydata\\rt-polarity.neg"
if __name__ == '__main__':
    x_data, y = data_reader(pos_path, neg_path)
    print(np.shape(x_data))
