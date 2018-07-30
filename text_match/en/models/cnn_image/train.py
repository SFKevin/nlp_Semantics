import tensorflow as tf
import numpy as np
from text_match.en.data_utils import datahelper
from tflearn.data_utils import pad_sequences
from text_match.en.models.cnn_image.CNN_model import cnn_text
import datetime
import os
import time

tf.flags.DEFINE_string("en_train", "I:\\CIKM\\cikm_english_train_20180516\\cikm_english_train_20180516.txt",
                       "en_train not found ")
tf.flags.DEFINE_string("sp_train", "I:\\CIKM\\cikm_spanish_train_20180516.txt",
                       "sp_train")
# tf.flags.DEFINE_string("en_train", "E:\\CIKM2018\\cikm_english_train_20180516\\cikm_english_train_20180516.txt",
#                        "en_train not found ")
# tf.flags.DEFINE_string("sp_train", "E:\\CIKM2018\\cikm_spanish_train_20180516.txt",
#                        "sp_train")
tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning_rate")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("hidden_size", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("clip_grad_up", 5.0, "clip_grad_up")
tf.flags.DEFINE_float("clip_grad_down", -5.0, "clip_grad_down")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
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
    x_text1, x_text2, y_train = datahelper.load_data(FLAGS.en_train, FLAGS.sp_train)
    x_text = np.concatenate([x_text1, x_text2], axis=0)
    word2index, index2word = datahelper.create_vocabulary(x_text)
    vocab_size = len(index2word)
    word_embedding = datahelper.asign_pretrained_word_embedding()

    max_len = max([len(x.split(" ")) for x in x_text])

    x_text1_int = []
    x_text2_int = []
    for line in x_text1:
        line_list = datahelper.text_to_wordlist(line)
        line_list = line_list.split(" ")
        text = [word2index.get(x, UNK_ID) for x in line_list]
        x_text1_int.append(text)

    for line in x_text2:
        line_list = datahelper.text_to_wordlist(line)
        line_list = line_list.split(" ")
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
            cnn = cnn_text(FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate,
                           max_len,
                           1, FLAGS.hidden_size, vocab_size, FLAGS.embedding_dim)

            learning_rate = tf.train.exponential_decay(cnn.learning_rate, cnn.global_step, cnn.decay_steps,
                                                       cnn.decay_rate, staircase=True)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss_val)

            grads_and_vars_clip = [(tf.clip_by_value(grad, FLAGS.clip_grad_down, FLAGS.clip_grad_up), var) for grad, var
                                   in
                                   grads_and_vars]

            train_op = optimizer.apply_gradients(grads_and_vars_clip, global_step=cnn.global_step)

            embedding = tf.constant(word_embedding, tf.float32)
            t_assign_embedding = tf.assign(cnn.Embedding, embedding)
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

            loss_summary = tf.summary.scalar("loss", cnn.loss_val)

            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            dev_summary_op = tf.summary.merge([loss_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            sess.run(tf.global_variables_initializer())

            checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "checkpoint", timestamp + "batch64"))
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

                feed_dict = {cnn.is_training: True, cnn.input_x1: x_batch1, cnn.input_x2: x_batch2,
                             cnn.input_y: y_batch,
                             cnn.dropout_keep_prob: FLAGS.dropout_keep_prob, cnn.weights: weights_t}
                summaries, _, step, loss = sess.run([train_summary_op, train_op, cnn.global_step, cnn.loss],
                                                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("Train: {}: step {}, loss {:g}. ".format(time_str, step, loss))
                log_write.write("Train: {}: step {}, loss {:g}.\n".format(time_str, step, loss))
                log_write.flush()
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch1, x_batch2, y_batch):
                total_loss = []
                step = 0
                test_batch = datahelper.batch_iter(list(zip(x_batch1, x_batch2, y_batch)), FLAGS.batch_size, 1)
                for test_data in test_batch:
                    x_dev_batch1, x_dev_batch2, y_dev_batch = zip(*test_data)
                    weights_d = []
                    for i in range(len(y_dev_batch)):
                        weights_d.append(1.0)
                    feed_dict = {cnn.is_training: False, cnn.input_x1: x_dev_batch1, cnn.input_x2: x_dev_batch2,
                                 cnn.input_y: y_dev_batch,
                                 cnn.dropout_keep_prob: 1.0, cnn.weights: weights_d}
                    summaries, step, loss = sess.run([dev_summary_op, cnn.global_step, cnn.loss], feed_dict)
                    total_loss.append(loss)
                loss = np.mean(total_loss, axis=0)
                time_str = datetime.datetime.now().isoformat()
                print("Test: {}:step {}, loss: {:g}".format(time_str, step, loss))
                log_write.write("Test: {}: step {}, loss {:g}.\n".format(time_str, step, loss))
                log_write.flush()
                dev_summary_writer.add_summary(summaries, step)

            batches = datahelper.batch_iter(list(zip(x_train1, x_train2, y_train)), FLAGS.batch_size,
                                            FLAGS.num_epochs)
            for batch in batches:
                x_batch1, x_batch2, y_batch = zip(*batch)
                train_step(x_batch1, x_batch2, y_batch)
                current_step = tf.train.global_step(sess, cnn.global_step)
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
