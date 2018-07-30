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
from sklearn.feature_extraction.text import CountVectorizer
import datetime

x_train1, x_train2, _ = datahelper.load_data(filepath_en_train, filepath_sp_train)

# x_train1 = process_data(x_train1)
# x_train2 = process_data(x_train2)
# x_test1 = process_data(x_test1)
# x_test2 = process_data(x_test2)

train = pd.DataFrame()
test = pd.DataFrame()

train['question1'] = x_train1
train['question2'] = x_train2

maxNumFeatures = 400

# bag of letter sequences (chars)
BagOfWordsExtractor = CountVectorizer(max_df=0.999, min_df=1000, max_features=maxNumFeatures,
                                      analyzer='char', ngram_range=(1, 3),
                                      binary=True, lowercase=True)

inputs = pd.concat(
    [train.ix[:, 'question1'], train.ix[:, 'question2']]).unique()
BagOfWordsExtractor.fit(inputs)


def makeFeature(df_features):
    now = datetime.datetime.now()
    print
    now.strftime('%Y-%m-%d %H:%M:%S')
    trainQuestion1_BOW_rep = BagOfWordsExtractor.transform(df_features.ix[:, 'question1'])
    trainQuestion2_BOW_rep = BagOfWordsExtractor.transform(df_features.ix[:, 'question2'])
    X = -(trainQuestion1_BOW_rep != trainQuestion2_BOW_rep).astype(int)
    df_features['f_bag_words'] = [X[i, :].toarray()[0] for i in range(0, len(df_features))]
    for j in range(0, len(df_features['f_bag_words'][0])):
        df_features['z_bag_words' + str(j)] = [df_features['f_bag_words'][i][j] for i in range(0, len(df_features))]
    df_features.fillna(0.0)
    now = datetime.datetime.now()
    print
    now.strftime('%Y-%m-%d %H:%M:%S')
    return df_features


if __name__ == "__main__":
    train = makeFeature(train)
    train.to_csv('train_bagofwords400.csv', index=False)
    print("done bag of words")
