import tensorflow as tf
import numpy as np
from CIKM.datautils import datahelper
from tflearn.data_utils import pad_sequences
from CIKM.models.mv_rnn.mvrnn_cos import MV_RNN
import datetime
import os
import time

tf.flags.DEFINE_string("en_train", "I:\\CIKM\\cikm_english_train_20180516\\cikm_english_train_20180516.txt",
                       "en_train not found ")
tf.flags.DEFINE_string("sp_train", "I:\\CIKM\\cikm_spanish_train_20180516.txt",
                       "sp_train")
tf.flags.DEFINE_string("stop_word", "I:\\CIKM\\spanish_stop_word.txt",
                       "stop_word")
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
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")
tf.flags.DEFINE_string("word2vec_model_path", "I:\\CIKM\\w2v.model.bin",
                       "Goole News Vector")
tf.flags.DEFINE_integer("decay_steps", 12000, "how many steps before deacy learning rate")
tf.flags.DEFINE_float("decay_rate", 0.9, "rate of decay for learning rate.")
FLAGS = tf.flags.FLAGS

PAD_ID = 0
UNK_ID = 1
_PAD = "_PAD"
_UNK = "UNK"


def process():
    x_text1, x_text2, _, y_train, _, x_train_reshape = datahelper.load_data(FLAGS.en_train, FLAGS.sp_train)
    word2index, index2word = datahelper.create_vocabulary(x_train_reshape)
    vocab_size = len(index2word)
    word_embedding = datahelper.asign_pretrained_word_embedding(index2word, vocab_size, FLAGS.word2vec_model_path)
    max_len = max([len(x.split(" ")) for x in x_train_reshape])

    x_text1_int = []
    x_text2_int = []
    stop_word = list(open(FLAGS.stop_word, "r", encoding='UTF-8').readlines())
    stop_word_list = [
        line.replace("\n", "").replace(",", "").replace(".", "").replace("?", "").replace("¿", "").replace("!",
                                                                                                           "").replace(
            "¡", "").lower() for
        line in
        stop_word]
    for line in x_text1:
        line_list = line.split(" ")
        line_list = [x for x in line_list if x not in stop_word_list]
        text = [word2index.get(x, UNK_ID) for x in line_list]
        x_text1_int.append(text)

    for line in x_text2:
        line_list = line.split(" ")
        line_list = [x for x in line_list if x not in stop_word_list]
        text = [word2index.get(x, UNK_ID) for x in line_list]
        x_text2_int.append(text)

    x_train1 = pad_sequences(x_text1_int, max_len)
    x_train2 = pad_sequences(x_text2_int, max_len)

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

    return x_train1, x_dev1, x_train2, x_dev2, y_train, y_dev, word_embedding, max_len, vocab_size


def train(x_train1, x_dev1, x_train2, x_dev2, y_train, y_dev, word_embedding, max_len, vocab_size):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            rnn = MV_RNN(FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate, max_len,
                         FLAGS.hidden_size, vocab_size, FLAGS.embedding_dim)

            learning_rate = tf.train.exponential_decay(rnn.learning_rate, rnn.global_step, rnn.decay_steps,
                                                       rnn.decay_rate, staircase=True)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads_and_vars = optimizer.compute_gradients(rnn.loss_val)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=rnn.global_step)

            embedding = tf.constant(word_embedding, tf.float32)
            t_assign_embedding = tf.assign(rnn.Embedding, embedding)
            sess.run(t_assign_embedding)

            # Keep track of gradient values and sparity
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

            loss_summary = tf.summary.scalar("loss", rnn.loss_val)

            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            dev_summary_op = tf.summary.merge([loss_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            sess.run(tf.global_variables_initializer())

            checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "checkpoint", timestamp + "_cos"))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            log_file = checkpoint_dir + "\\log.txt"
            log_write = open(log_file, 'w', encoding='utf-8')
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)

            def train_step(x_batch1, x_batch2, y_batch):
                weights_t = []
                for i in range(len(y_batch)):
                    if y_batch[i] == 1:
                        weights_t.append(1.0)
                    else:
                        weights_t.append(1)

                feed_dict = {rnn.is_training: True, rnn.input_x1: x_batch1, rnn.input_x2: x_batch2,
                             rnn.input_y: y_batch,
                             rnn.dropout_keep_prob: FLAGS.dropout_keep_prob, rnn.weights: weights_t}
                summaries, _, step, loss = sess.run([train_summary_op, train_op, rnn.global_step, rnn.loss_val],
                                                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("Train: {}: step {}, loss {:g}. ".format(time_str, step, loss))
                tmp = "Train: {}: step {}, loss {:g}.\n".format(time_str, step, loss)
                log_write.write(tmp)
                log_write.flush()
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch1, x_batch2, y_batch):
                weights_d = []
                for i in range(len(y_batch)):
                    weights_d.append(1)
                feed_dict = {rnn.is_training: False, rnn.input_x1: x_batch1, rnn.input_x2: x_batch2,
                             rnn.input_y: y_batch,
                             rnn.dropout_keep_prob: 1.0, rnn.weights: weights_d}
                summaries, step, loss = sess.run([dev_summary_op, rnn.global_step, rnn.loss_val], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("Test: {}:step {}, loss: {:g}".format(time_str, step, loss))
                log_write.write("Test: {}: step {}, loss {:g}.\n".format(time_str, step, loss))
                log_write.flush()
                dev_summary_writer.add_summary(summaries, step)

            batches = datahelper.batch_iter(list(zip(x_train1, x_train2, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            for batch in batches:
                x_batch1, x_batch2, y_batch = zip(*batch)
                train_step(x_batch1, x_batch2, y_batch)
                current_step = tf.train.global_step(sess, rnn.global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev1, x_dev2, y_dev)
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
            log_write.close()


if __name__ == '__main__':
    x_train1, x_dev1, x_train2, x_dev2, y_train, y_dev, word_embedding, max_len, vocab_size = process()
    train(x_train1, x_dev1, x_train2, x_dev2, y_train, y_dev, word_embedding, max_len, vocab_size)
