import tensorflow as tf
import numpy as np
from CIKM.datautils import datahelper
from tflearn.data_utils import pad_sequences
import os
import time

tf.flags.DEFINE_string("en_train", "I:\\CIKM\\cikm_english_train_20180516\\cikm_english_train_20180516.txt",
                       "en_train not found ")
tf.flags.DEFINE_string("sp_train", "I:\\CIKM\\cikm_spanish_train_20180516.txt",
                       "sp_train")
tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning_rate")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("hidden_size", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 2000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("word2vec_model_path", "I:\\CIKM\\w2v.model.bin",
                       "Goole News Vector")
tf.flags.DEFINE_integer("decay_steps", 12000, "how many steps before deacy learning rate")
tf.flags.DEFINE_float("decay_rate", 0.9, "rate of decay for learning rate.")
FLAGS = tf.flags.FLAGS

PAD_ID = 0
UNK_ID = 1
_PAD = "_PAD"
_UNK = "UNK"
filepath_test = "I:\\CIKM\\cikm_test_a_20180516.txt"


def process():
    x_text1, x_text2, _, y_train, _, x_train_reshape = datahelper.load_data(FLAGS.en_train, FLAGS.sp_train)
    word2index, index2word = datahelper.create_vocabulary(x_train_reshape)
    vocab_size = len(index2word)
    word_embedding = datahelper.asign_pretrained_word_embedding(index2word, vocab_size, FLAGS.word2vec_model_path)
    max_len = max([len(x.split(" ")) for x in x_train_reshape])
    test1, test2 = datahelper.load_testdata(filepath_test)
    test1_int = []
    test2_int = []

    x_text1_int = []
    x_text2_int = []

    for line in x_text1:
        line_list = line.split(" ")
        text = [word2index.get(x, UNK_ID) for x in line_list]
        x_text1_int.append(text)

    for line in x_text2:
        line_list = line.split(" ")
        text = [word2index.get(x, UNK_ID) for x in line_list]
        x_text2_int.append(text)

    for line in test1:
        line_list = line.split(" ")
        text = [word2index.get(x, UNK_ID) for x in line_list]
        test1_int.append(text)

    for line in test2:
        line_list = line.split(" ")
        text = [word2index.get(x, UNK_ID) for x in line_list]
        test2_int.append(text)

    x_train1 = pad_sequences(x_text1_int, max_len)
    x_train2 = pad_sequences(x_text2_int, max_len)
    x_test1 = pad_sequences(test1_int, max_len)
    x_test2 = pad_sequences(test2_int, max_len)

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_shuffled1 = x_train1[shuffle_indices]
    x_shuffled2 = x_train2[shuffle_indices]
    y_shuffled = y_train[shuffle_indices]

    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_train)))
    x_train1, x_dev1 = x_shuffled1[:dev_sample_index], x_shuffled1[dev_sample_index:]
    x_train2, x_dev2 = x_shuffled2[:dev_sample_index], x_shuffled2[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x_text1, x_text2, x_text1_int, x_text2_int

    return x_shuffled1, x_shuffled2, x_train2, x_dev2, y_train, y_dev, word_embedding, max_len, vocab_size, x_test1, x_test2


def predict(x_pre1, x_pre2):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            x_test1 = x_pre1  # tf.nn.embedding_lookup(word_embedding, x_pre1)
            x_test2 = x_pre2  # tf.nn.embedding_lookup(word_embedding, x_pre2)
            checkpoint_dir = os.path.abspath(
                os.path.join(os.path.curdir, "checkpoint", "1532876361_bn"))
            model_path = os.path.join(checkpoint_dir, "model-15100.meta")
            saver = tf.train.import_meta_graph(model_path)
            if os.path.exists(checkpoint_dir):
                print("Restoring Variables from Checkpoint for rnn")
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            else:
                print("Can't find the checkpoint.going to stop")
                return
            graph = tf.get_default_graph()
            batches = datahelper.batch_iter(list(zip(x_test1, x_test2)), FLAGS.batch_size,
                                            1, shuffle=False)
            result_list = []
            for test_data in batches:
                x_dev_batch1, x_dev_batch2 = zip(*test_data)
                weights = []
                for i in range(len(x_dev_batch1)):
                    weights.append(1.0)
                result = graph.get_tensor_by_name("outputs/predict:0")
                logits = sess.run(result,
                                  {"is_training:0": False, "input_x1:0": x_dev_batch1, "input_x2:0": x_dev_batch2,
                                   "dropout_keep_prob:0": 1.0, "weights:0": weights})
                result_list = np.concatenate([np.array(result_list), np.array(logits)], axis=0)
        return np.array(result_list)


if __name__ == '__main__':
    x_train1, x_train2, _, _, _, _, word_embedding, max_len, vocab_size, x_test1, x_test2 = process()
    result = predict(x_train1, x_train2)
    timestamp = str(int(time.time()))
    file_name = timestamp + "train_HAN.txt"
    np.savetxt(file_name, result)
    print("number of result: %d" % len(result))
