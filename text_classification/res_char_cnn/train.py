import tensorflow as tf
import numpy as np
from res_char_cnn import datautils
from res_char_cnn.ResModel import ResNet
import time
import os
import datetime

tf.flags.DEFINE_string("positive_data_file", "I:\\data\\rt-polaritydata\\rt-polarity.pos",
                       "positive_data_file not found ")
tf.flags.DEFINE_string("negative_data_file", "I:\\data\\rt-polaritydata\\rt-polarity.neg",
                       "negative_data_file not found")
tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 0, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("word2vec_model_path", "../GoogleNews-vectors-negative300.bin", "Goole News Vector")
FLAGS = tf.flags.FLAGS


def preprocess():
    x_text, y, maxlen = datautils.data_reader(
        FLAGS.positive_data_file, FLAGS.negative_data_file)

    # Random shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = np.array(x_text)[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    del x_text, y, x_shuffled, y_shuffled

    return x_train, y_train, x_dev, y_dev, maxlen


def train(x_train, y_train, x_dev, y_dev, maxlen):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            res_cnn = ResNet(l0=maxlen, num_class=y_train.shape[1], l2_reg_lambda=FLAGS.l2_reg_lambda)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(res_cnn.loss)
            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram(
                        "{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar(
                        "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(
                os.path.curdir, "runs_new", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", res_cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", res_cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge(
                [loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(
                train_summary_dir, sess.graph)
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    res_cnn.input_x: x_batch,
                    res_cnn.input_y: y_batch,
                    res_cnn.training: True,
                    res_cnn.dropout_keep_prob: 0.5
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step,
                     res_cnn.loss, res_cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                # write fewer training summaries, to keep events file from
                # growing so big.
                print("{}: step {}, loss {:g}, acc {:g}".format(
                    time_str, step, loss, accuracy))

            def dev_step(x_batch, y_batch):
                feed_dict = {res_cnn.input_x: x_batch, res_cnn.input_y: y_batch,
                             res_cnn.training: False, res_cnn.dropout_keep_prob: 1.0}
                step, loss, accuracy = sess.run(
                    [global_step, res_cnn.loss, res_cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}:step {}, loss: {:g}, acc: {:g}".format(time_str, step, loss, accuracy))

            batches = datautils.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev)
                    print("")


if __name__ == '__main__':
    x_train, y_train, x_dev, y_dev, maxlen = preprocess()
    train(x_train, y_train, x_dev, y_dev, maxlen)
