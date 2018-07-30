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

x_train1, x_train2, _ = datahelper.load_data(filepath_en_train, filepath_sp_train)

train = pd.DataFrame()

train['question1'] = x_train1
train['question2'] = x_train2

df1 = train[['question1']].copy()
df2 = train[['question2']].copy()

df2.rename(columns={'question2': 'question1'}, inplace=True)

train_questions = df1.append(df2)
train_questions.drop_duplicates(['question1'], inplace=True)

train_questions.reset_index(inplace=True, drop=True)
questions_dict = pd.Series(train_questions.index.values, index=train_questions.question1.values).to_dict()
train_cp = train.copy()

train_cp['is_duplicate'] = 1

comb = train_cp

comb['q1_hash'] = comb['question1'].map(questions_dict)
comb['q2_hash'] = comb['question2'].map(questions_dict)

q1_vc = comb.q1_hash.value_counts().to_dict()
q2_vc = comb.q2_hash.value_counts().to_dict()


def try_apply_dict(x, dict_to_apply):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0


comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x, q1_vc) + try_apply_dict(x, q2_vc))
comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x, q1_vc) + try_apply_dict(x, q2_vc))

train_comb = comb[comb['is_duplicate'] >= 0][['q1_hash', 'q2_hash', 'q1_freq', 'q2_freq']]

train_comb.to_csv('train_freq.csv', index=False)
