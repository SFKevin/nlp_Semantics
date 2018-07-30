import tensorflow as tf
from tensorflow.contrib import rnn


class TextRNN:
    def __init__(self, num_classes, learning_rate, batch_size, decay_step, decay_rate, sequence_length,
                 vocab_size, embed_size, is_training, initializer=tf.random_normal_initializer(stddev=0.1)):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = embed_size
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.initializer = initializer

        # add placeholder
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None,self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="drop_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_step, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()
        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
        correct_prediction = tf.equal(self.predictions, tf.arg_max(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam")
        return train_op

    def instantiate_weights(self):
        with tf.name_scope("embedding"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 2, self.num_classes],
                                                initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

    def inference(self):
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)

        if self.dropout_keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words,
                                                         dtype=tf.float32)
            outputs_rnn = tf.concat(outputs, axis=2)
            self.output_rnn_last = outputs_rnn[:, -1, :]
            with tf.name_scope("output"):
                logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection
            return logits

    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.nn.l2_loss(self.W_projection)+tf.nn.l2_loss(self.b_projection)
            loss = loss + l2_lambda*l2_losses
            return loss
