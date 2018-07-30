import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
filepath_pos="E:\\data\\rt-polaritydata\\rt-polarity.pos"
filepath_neg="E:\\data\\rt-polaritydata\\rt-polarity.neg"
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
    x_text = [clean_str(sent) for sent in x_text]
    # x_text=[x.split(" ") for x in x_text]
    tfidf=TfidfVectorizer(ngram_range=[1,5],stop_words=["a","an","the"],norm="l2",max_df=0.7,min_df=0.0005)
    x_tfidf=tfidf.fit_transform(x_text).toarray()
    # labels
    positive_labels = [1 for _ in positive_examples]
    negative_lables = [0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_lables], 0)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x_tfidf[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split
    dev_sample_index = -1 * int(0.1 * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    return x_train,y_train,x_dev,y_dev

if __name__ == '__main__':
    x_train,y_train,x_dev,y_dev=load_data_and_labels(filepath_pos,filepath_neg)
    print(np.shape(x_train))
    print(np.shape(y_train))