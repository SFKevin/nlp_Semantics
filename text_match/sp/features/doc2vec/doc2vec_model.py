from CIKM.datautils import datahelper
import gensim
import time
import datetime
from collections import namedtuple

# filepath_en_train = "E:\\CIKM2018\\cikm_english_train_20180516\\cikm_english_train_20180516.txt"
# filepath_sp_train = "E:\\CIKM2018\\cikm_spanish_train_20180516.txt"
# filepath_test = "E:\\CIKM2018\\cikm_test_a_20180516.txt"
# filepath_unlabel = "E:\\CIKM2018\\cikm_unlabel_spanish_train_20180516\\cikm_unlabel_spanish_train_20180516.txt"
# filepath_stop_word = "E:\\CIKM2018\\spanish_stop_word.txt"
filepath_en_train = "I:\\CIKM\\cikm_english_train_20180516\\cikm_english_train_20180516.txt"
filepath_sp_train = "I:\\CIKM\\cikm_spanish_train_20180516.txt"
filepath_test = "I:\\CIKM\\cikm_test_a_20180516.txt"
filepath_unlabel = "I:\\CIKM\\cikm_unlabel_spanish_train_20180516\\cikm_unlabel_spanish_train_20180516.txt"
w2v_pah = "I:\\CIKM\\w2v.model.bin"
fast_path = "I:\\CIKM\\fast_text_vectors_wiki.es.vec\\wiki.es.vec"
file_stop_word = "I:\\CIKM\\spanish_stop_word.txt"


def process():
    _, _, _, _, w2v_list, _ = datahelper.load_data(filepath_en_train, filepath_sp_train)

    stop_word = list(open(file_stop_word, "r", encoding='UTF-8').readlines())
    stop_word_list = [
        line.replace("\n", "").replace(",", "").replace(".", "").replace("?", "").replace("¿", "").replace("!",
                                                                                                           "").replace(
            "¡", "").lower() for
        line in
        stop_word]
    d2c_list = []
    for line in w2v_list:
        # line_list = [x for x in line if x not in stop_word_list]
        d2c_list.append(line)

    alldocuments = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for id, record in enumerate(d2c_list):
        qid = str('SENT_%s' % id)
        words = record
        words_text = " ".join(words)
        words = gensim.utils.simple_preprocess(words_text)
        tags = [qid]
        alldocuments.append(analyzedDocument(words, tags))
    print("Start Training Doc2Vec Time : %s" % (str(datetime.datetime.now())))
    saved_model_name = "doc_2_vec_" + str(int(time.time()))
    model_4 = gensim.models.Doc2Vec(alldocuments, dm=1, dm_concat=1, vector_size=300, window=5,
                                    min_count=2, epochs=100)
    model_4.save("%s" % (saved_model_name))
    print("model training completed : %s" % (saved_model_name))


def get_question_vector(question1, model):
    # model_name = "%s" % ("doc_2_vec_1531711452")
    # model_saved_file = "%s" % (model_name)
    # model = gensim.models.doc2vec.Doc2Vec.load(model_saved_file)
    tokenize_text1 = ' '.join(question1)
    tokenize_text1 = gensim.utils.simple_preprocess(tokenize_text1)
    infer_vector_of_question1 = model.infer_vector(tokenize_text1)
    return infer_vector_of_question1


# <class 'list'>: ['hola', 'hago', 'clic', 'en', 'el', 'producto', 'recibido']
if __name__ == '__main__':
    process()
    # example = ['hola', 'hago', 'clic', 'en', 'el', 'producto', 'recibido']
    # print(get_question_vector(example))
