import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


class rnn_dot:
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
        self.predictions = self.logits
        self.loss_val = self.loss()

    def instantiate_weights(self):
        with tf.name_scope("Embedding"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer, trainable=True)

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
        self.image = tf.matmul(self.outputs1_rnn, self.outputs2_rnn, transpose_b=True)
        self.image_exp = tf.expand_dims(self.image, axis=-1)

        with tf.variable_scope("conv1"):
            self.conv1 = tf.layers.conv2d(inputs=self.image_exp, filters=128, kernel_size=3, strides=[1, 1],
                                          padding="SAME"
                                          , use_bias=True,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv")
            self.bn1 = tf.layers.batch_normalization(inputs=self.conv1, axis=3, momentum=_BATCH_NORM_DECAY,
                                                     epsilon=_BATCH_NORM_EPSILON, center=True,
                                                     scale=True, training=self.is_training, fused=True)
            self.activ1 = tf.nn.relu(self.bn1)

            self.pool1 = tf.layers.average_pooling2d(self.activ1, pool_size=2, strides=[2, 2], padding="SAME",
                                                     name="pool")

        with tf.variable_scope("conv2"):
            self.conv2 = tf.layers.conv2d(inputs=self.pool1, filters=128, kernel_size=3, strides=[1, 1], padding="SAME",
                                          use_bias=True,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv")
            self.bn2 = tf.layers.batch_normalization(inputs=self.conv2, axis=3, momentum=_BATCH_NORM_DECAY,
                                                     epsilon=_BATCH_NORM_EPSILON, center=True,
                                                     scale=True, training=self.is_training, fused=True)
            self.activ2 = tf.nn.relu(self.bn2)

            self.pool2 = tf.layers.average_pooling2d(self.activ2, pool_size=2, strides=[2, 2], padding="SAME",
                                                     name="pool")
        with tf.variable_scope("conv3"):
            self.conv3 = tf.layers.conv2d(inputs=self.pool2, filters=256, kernel_size=3, strides=[1, 1], padding="SAME",
                                          use_bias=True,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv")
            self.bn3 = tf.layers.batch_normalization(inputs=self.conv3, axis=3, momentum=_BATCH_NORM_DECAY,
                                                     epsilon=_BATCH_NORM_EPSILON, center=True,
                                                     scale=True, training=self.is_training, fused=True)
            self.activ3 = tf.nn.relu(self.bn3)

            self.pool3 = tf.keras.layers.GlobalAveragePooling2D()(self.activ3)
        with tf.variable_scope("outputs"):
            # self.manha = tf.reduce_sum(tf.abs(tf.subtract(self.outputs1_last, self.outputs2_last)), axis=1,
            #                            keepdims=True)
            # self.squre = tf.reduce_sum(tf.square(tf.subtract(self.outputs1_last, self.outputs2_last)), axis=1,
            #                            keepdims=True)
            #
            # self.element_wise = tf.multiply(self.outputs1_last, self.outputs2_last)
            # self.norm1 = tf.sqrt(tf.reduce_sum(tf.square(self.outputs1_last), axis=1, keepdims=True))
            # self.norm2 = tf.sqrt(tf.reduce_sum(tf.square(self.outputs2_last), axis=1, keepdims=True))
            # self.sum12 = tf.reduce_sum(self.element_wise, axis=1, keepdims=True)
            # self.cos = tf.divide(self.sum12, tf.multiply(self.norm1, self.norm2))
            #
            # self.input_dense = tf.concat(
            #     [self.element_wise, self.manha, self.norm1, self.norm2, self.sum12, self.cos, self.squre], axis=1)
            # self.input_dense_norm = tf.layers.batch_normalization(self.input_dense, training=self.is_training)
            #
            self.fc1 = tf.layers.dense(self.pool3, 256, activation=tf.nn.leaky_relu)
            self.fc1 = tf.nn.dropout(self.fc1, keep_prob=self.dropout_keep_prob)

            self.fc2 = tf.layers.dense(self.fc1, 128, activation=tf.nn.leaky_relu)
            self.fc2 = tf.nn.dropout(self.fc2, keep_prob=self.dropout_keep_prob)

            self.fc3 = tf.layers.dense(self.fc2, 32, activation=tf.nn.leaky_relu)
            self.fc3 = tf.nn.dropout(self.fc3, keep_prob=self.dropout_keep_prob)
            self.logits = tf.squeeze(tf.layers.dense(self.fc3, 1, activation=tf.nn.sigmoid), axis=1,
                                     name="predict")
        return self.logits

    def loss(self):
        self.loss = tf.losses.log_loss(labels=self.input_y, predictions=self.logits,
                                       reduction="weighted_mean")
        self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 0.01
        return self.loss + self.l2_losses
