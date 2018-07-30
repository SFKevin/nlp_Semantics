import datetime
import tensorflow as tf
import numpy as np

from datapre import datahelper
from HAN.HAN_model import HierarchicalAttention
from tflearn.data_utils import pad_sequences

tf.flags.DEFINE_string("positive_data_file", "I:\\data\\rt-polaritydata\\rt-polarity.pos", "positive_data_file not found ")
tf.flags.DEFINE_string("negative_data_file", "I:\\data\\rt-polaritydata\\rt-polarity.neg", "negative_data_file not found")
tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation")
# Model Hyperparameters
tf.flags.DEFINE_float("num_classes", 2, "number of class")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("decay_steps", 12000, "how many steps before deacy learning rate")
tf.flags.DEFINE_float("decay_rate", 0.9, "rate of decay for learning rate.")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("word2vec_model_path", "I:\\data\\GoogleNews-vectors-negative300.bin", "Goole News Vector")
FLAGS = tf.flags.FLAGS

PAD_ID = 0
UNK_ID = 1
_PAD = "_PAD"
_UNK = "UNK"

def process():
    x_text, y = datahelper.load_data_and_labels(
        FLAGS.positive_data_file, FLAGS.negative_data_file)
    vocabulary_word2index, vocabulary_index2word = datahelper.create_vocabulary(x_text)
    vocab_size = len(vocabulary_word2index)
    # word_embedding = datahelper.asign_pretrained_word_embedding(vocabulary_index2word, vocab_size,
    #                                                                     FLAGS.word2vec_model_path)
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    x_int_text = []
    for line in x_text:
        input_list = line.split(" ")
        text = [vocabulary_word2index.get(x, UNK_ID) for x in input_list]
        x_int_text.append(text)

    x = pad_sequences(x_int_text, maxlen=max_document_length, value=0.0)

    # Random shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    del x, y, x_text, x_shuffled, y_shuffled

    return x_train, y_train, vocab_size, x_dev, y_dev#, word_embedding
def train(x_train,y_train,vocab_size,x_dev,y_dev):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            HANmodel=HierarchicalAttention(FLAGS.num_classes,0.01,FLAGS.batch_size,FLAGS.decay_steps,FLAGS.decay_rate
                                           ,59,1,vocab_size,FLAGS.embedding_dim,FLAGS.embedding_dim,True)
            # embedding=tf.constant(word_embeddings,tf.float32)
            # t_assign_embedding=tf.assign(HANmodel.Embedding,embedding)
            # sess.run(t_assign_embedding)
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch,y_batch):
                feed_dict = {HANmodel.input_x: x_batch, HANmodel.input_y: y_batch,
                             HANmodel.dropout_keep_prob: 0.5}
                _,step,loss,accuracy=sess.run([HANmodel.train_op,HANmodel.global_step,HANmodel.loss_val,HANmodel.accuracy],feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}. acc {:g}".format(time_str, step, loss, accuracy))

            def dev_step(x_batch, y_batch, writer=None):
                feed_dict = {HANmodel.input_x: x_batch, HANmodel.input_y: y_batch, HANmodel.dropout_keep_prob: 1.0}
                step, loss, accuracy = sess.run([HANmodel.global_step, HANmodel.loss_val, HANmodel.accuracy],
                                                           feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}:step {}, loss: {:g}, acc: {:g}".format(time_str, step, loss, accuracy))

            batches = datahelper.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, HANmodel.global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev)
                    print("")

if __name__ == '__main__':
    x_train, y_train, vocab_size, x_dev, y_dev = process()
    train(x_train, y_train, vocab_size, x_dev, y_dev)