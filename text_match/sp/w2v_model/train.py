import gensim
from CIKM.datautils import datahelper

filepath_en_train = "E:\\CIKM2018\\cikm_english_train_20180516\\cikm_english_train_20180516.txt"
filepath_sp_train = "E:\\CIKM2018\\cikm_spanish_train_20180516.txt"
filepath_test = "E:\\CIKM2018\\cikm_test_a_20180516.txt"
filepath_unlabel = "E:\\CIKM2018\\cikm_unlabel_spanish_train_20180516\\cikm_unlabel_spanish_train_20180516.txt"
w2v_pah = "E:\\CIKM2018\\w2v.model.bin"


def train_w2v():
    _, _, _, _, data, _ = datahelper.load_data(filepath_en_train, filepath_sp_train)
    model = gensim.models.Word2Vec(data, size=300, min_count=1)
    model.wv.save_word2vec_format('w2v.model.bin', binary=True)


if __name__ == '__main__':
    # model=gensim.models.Word2Vec.load("w2v_model")
    # print(model['podido'])
    train_w2v()
