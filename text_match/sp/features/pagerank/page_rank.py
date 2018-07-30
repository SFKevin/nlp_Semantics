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
import hashlib
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


# Generating a graph of Questions and their neighbors
def generate_qid_graph_table(row):
    hash_key1 = hashlib.md5(row["question1"].encode('utf-8')).hexdigest()
    hash_key2 = hashlib.md5(row["question2"].encode('utf-8')).hexdigest()

    qid_graph.setdefault(hash_key1, []).append(hash_key2)
    qid_graph.setdefault(hash_key2, []).append(hash_key1)


qid_graph = {}
print('Apply to train...')
train.apply(generate_qid_graph_table, axis=1)
print('Apply to test...')
test.apply(generate_qid_graph_table, axis=1)


def pagerank():
    MAX_ITER = 20
    d = 0.85

    # Initializing -- every node gets a uniform value!
    pagerank_dict = {i: 1 / len(qid_graph) for i in qid_graph}
    num_nodes = len(pagerank_dict)

    for iter in range(0, MAX_ITER):

        for node in qid_graph:
            local_pr = 0

            for neighbor in qid_graph[node]:
                local_pr += pagerank_dict[neighbor] / len(qid_graph[neighbor])

            pagerank_dict[node] = (1 - d) / num_nodes + d * local_pr

    return pagerank_dict


print('Main PR generator...')
pagerank_dict = pagerank()


def get_pagerank_value(row):
    try:
        q1 = hashlib.md5(row["question1"].encode('utf-8')).hexdigest()
        q2 = hashlib.md5(row["question2"].encode('utf-8')).hexdigest()
    except:
        print(hashlib.md5(row["question1"].encode('utf-8')).hexdigest())
        print(hashlib.md5(row["question2"].encode('utf-8')).hexdigest())
    s = pd.Series({
        "f_q1_pr": pagerank_dict[q1],
        "f_q2_pr": pagerank_dict[q2]
    })

    return s


print('Apply to train...')
pagerank_feats_train = train.apply(get_pagerank_value, axis=1)
print('Writing train...')
pagerank_feats_train.to_csv("pagerank_train.csv", index=False)
print('Apply to test...')
pagerank_feats_test = test.apply(get_pagerank_value, axis=1)
print('Writing test...')
pagerank_feats_test.to_csv("pagerank_test.csv", index=False)
