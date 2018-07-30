import tensorflow as tf


class MV_RNN:
    def __init__(self, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 hidden_size, vocab_size, embedding_size,
                 initializer=tf.random_normal_initializer(stddev=0.1), clip_gradients=5.0):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5)
        self.initializer = initializer
        self.hidden_size = hidden_size
        self.clip_gradients = clip_gradients
        self.vocab_size = vocab_size
        self.embed_size = embedding_size

        self.input_x1 = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x2")

        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.weights = tf.placeholder(tf.float32, [None], name="weights")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()
        self.loss_val = self.loss()

    def inference(self):
        self.input_x1_embed = tf.nn.embedding_lookup(self.Embedding, self.input_x1)
        self.input_x2_embed = tf.nn.embedding_lookup(self.Embedding, self.input_x2)

        with tf.variable_scope("lstm") as scope:
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
            outputs1, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.input_x1_embed,
                                                          dtype=tf.float32)
            self.outputs1_rnn = tf.concat(outputs1, axis=2)

            scope.reuse_variables()
            outputs2, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.input_x2_embed,
                                                          dtype=tf.float32)

            self.outputs2_rnn = tf.concat(outputs2, axis=2)

        with tf.variable_scope("cos"):
            self.dot_wise = tf.matmul(self.outputs1_rnn, self.outputs2_rnn, transpose_b=True, name="matual")
            self.norm1 = tf.sqrt(tf.reduce_sum(tf.square(self.outputs1_rnn), axis=2, keepdims=True), name="norm1")
            self.norm2 = tf.sqrt(tf.reduce_sum(tf.square(self.outputs2_rnn), axis=2, keepdims=True), name="norm1")
            self.norm = tf.matmul(self.norm1, self.norm2, transpose_b=True, name="matual_norm")
            self.cos = tf.div(self.dot_wise, self.norm, name="cos")
            self.kmax = tf.nn.top_k(self.cos, k=8, name="k-max-pool")
            self.inputs = tf.layers.flatten(self.kmax[0], name="flatten")
        with tf.variable_scope("outputs"):
            self.fc1 = tf.layers.dense(self.inputs, 256, activation=tf.nn.relu)
            self.fc1 = tf.nn.dropout(self.fc1, keep_prob=self.dropout_keep_prob)

            self.fc2 = tf.layers.dense(self.fc1, 128, activation=tf.nn.relu)
            self.fc2 = tf.nn.dropout(self.fc2, keep_prob=self.dropout_keep_prob)

            self.fc3 = tf.layers.dense(self.fc2, 32, activation=tf.nn.relu)
            self.fc3 = tf.nn.dropout(self.fc3, keep_prob=self.dropout_keep_prob)

            self.logits = tf.squeeze(tf.layers.dense(self.fc3, 1, activation=tf.nn.sigmoid), axis=1, name="predict")
        return self.logits

    def loss(self):
        self.loss = tf.losses.log_loss(labels=self.input_y, predictions=self.logits, weights=self.weights,
                                       reduction="weighted_mean")
        self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "outputs" in v.name]) * 0.0001
        return self.loss + self.l2_losses

    def instantiate_weights(self):
        with tf.name_scope("Embedding"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer, trainable=False)
