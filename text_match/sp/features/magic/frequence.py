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

df1 = train[['question1']].copy()
df2 = train[['question2']].copy()
df1_test = test[['question1']].copy()
df2_test = test[['question2']].copy()

df2.rename(columns={'question2': 'question1'}, inplace=True)
df2_test.rename(columns={'question2': 'question1'}, inplace=True)

train_questions = df1.append(df2)
train_questions = train_questions.append(df1_test)
train_questions = train_questions.append(df2_test)
train_questions.drop_duplicates(['question1'], inplace=True)

train_questions.reset_index(inplace=True, drop=True)
questions_dict = pd.Series(train_questions.index.values, index=train_questions.question1.values).to_dict()
train_cp = train.copy()
test_cp = test.copy()

test_cp['is_duplicate'] = -1
train_cp['is_duplicate'] = 1

comb = pd.concat([train_cp, test_cp])

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
test_comb = comb[comb['is_duplicate'] < 0][['q1_hash', 'q2_hash', 'q1_freq', 'q2_freq']]

train_comb.to_csv('train_freq.csv', index=False)
test_comb.to_csv('test_freq.csv', index=False)
