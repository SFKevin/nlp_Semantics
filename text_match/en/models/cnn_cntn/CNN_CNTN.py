import tensorflow as tf


class cnn_cntn:
    def __init__(self, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 hidden_size, vocab_size, embed_size,
                 initializer=tf.random_normal_initializer(stddev=0.1), clip_gradients=5.0):  # 0.01
        """init all hyperparameter here"""
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5)
        self.initializer = initializer
        self.hidden_size = hidden_size
        self.clip_gradients = clip_gradients
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.input_x1 = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x2")

        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.weights = tf.placeholder(tf.float32, [None], name="weights")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_step")
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        self.instantiate_weights()
        self.logits = self.inference()
        self.loss_val = self.loss()

    def inference(self):
        self.input_x1_embed = tf.nn.embedding_lookup(self.Embedding, self.input_x1)
        self.input_x2_embed = tf.nn.embedding_lookup(self.Embedding, self.input_x2)
        self.input_x1_embed_exp = tf.expand_dims(self.input_x1_embed, axis=-1)
        self.input_x2_embed_exp = tf.expand_dims(self.input_x2_embed, axis=-1)

        with tf.variable_scope("conv1") as scope:
            self.conv11 = tf.layers.conv2d(self.input_x1_embed_exp, filters=128, kernel_size=[3, self.embed_size],
                                           strides=[1, 1],
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='VALID',
                                           name="conv1")
            self.activ11 = tf.nn.relu(self.conv11, name="relu1")
            self.pool11 = tf.layers.max_pooling2d(self.activ11, pool_size=[2, 1], strides=[2, 1], padding='SAME',
                                                  name="pool11")
            scope.reuse_variables()
            self.conv12 = tf.layers.conv2d(self.input_x2_embed_exp, filters=128, kernel_size=[3, self.embed_size],
                                           strides=[1, 1],
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='VALID',
                                           name="conv1")
            self.activ12 = tf.nn.relu(self.conv12, name="relu1")
            self.pool12 = tf.layers.max_pooling2d(self.activ12, pool_size=[2, 1], strides=[2, 1], padding='SAME',
                                                  name="pool12")
        with tf.variable_scope("conv2") as scope:
            self.conv21 = tf.layers.conv2d(self.pool11, filters=128, kernel_size=[3, 1], strides=[1, 1], padding='SAME',
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv2")
            self.activ21 = tf.nn.relu(self.conv21, name="relu2")
            self.pool21 = tf.layers.max_pooling2d(self.activ21, pool_size=[2, 1], strides=[2, 1], name="pool21")
            scope.reuse_variables()
            self.conv22 = tf.layers.conv2d(self.pool12, filters=128, kernel_size=[3, 1], strides=[1, 1], padding='SAME',
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv2")
            self.activ22 = tf.nn.relu(self.conv22, name="relu2")
            self.pool22 = tf.layers.max_pooling2d(self.activ22, pool_size=[2, 1], strides=[2, 1], padding='SAME',
                                                  name="pool22")
        with tf.variable_scope("conv3") as scope:
            self.conv31 = tf.layers.conv2d(self.pool21, filters=128, kernel_size=[3, 1], strides=[1, 1], padding='SAME',
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv3")
            self.bn31 = tf.layers.batch_normalization(self.conv31, name="bn3")
            self.activ31 = tf.nn.relu(self.bn31)
            self.tranp1 = tf.transpose(self.activ31, [0, 3, 2, 1])
            self.kmax1 = tf.nn.top_k(self.tranp1, k=8)
            self.input1 = tf.layers.flatten(self.kmax1[0])
            self.input1 = tf.expand_dims(self.input1, axis=1)
            scope.reuse_variables()
            self.conv32 = tf.layers.conv2d(self.pool22, filters=128, kernel_size=[3, 1], strides=[1, 1], padding='SAME',
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv3")
            self.bn32 = tf.layers.batch_normalization(self.conv32, name="bn3")
            self.activ32 = tf.nn.relu(self.bn32)
            self.tranp2 = tf.transpose(self.activ32, [0, 3, 2, 1])
            self.kmax2 = tf.nn.top_k(self.tranp2, k=8)
            self.input2 = tf.layers.flatten(self.kmax2[0])
            self.input2 = tf.expand_dims(self.input2, axis=1)
        self.length = self.input1.get_shape().as_list()[-1]
        self.W1 = tf.get_variable("W1", shape=[self.length, 3, self.length], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.V1 = tf.get_variable("V1", shape=[2 * self.length, 3], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable(name="bias", shape=[3], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        self.U1 = tf.get_variable(name="U", shape=[3, 1], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.concat12 = tf.concat([self.input1, self.input2], axis=2)
        self.f0 = tf.einsum("abc,cd->abd", self.concat12, self.V1)
        self.f0_add = tf.add(self.f0, self.b)
        self.f1 = tf.einsum("abc,cde->abde", self.input1, self.W1)
        self.f1_sq = tf.squeeze(self.f1, axis=1)
        self.f2 = tf.matmul(self.f1_sq, self.input2, transpose_b=True)
        self.f2_tranp = tf.transpose(self.f2, [0, 2, 1])
        self.f2_tranp = tf.nn.relu(self.f2_tranp)
        with tf.variable_scope("outputs"):
            self.input_tmp = tf.add(self.f0_add, self.f2_tranp)
            self.inputs = tf.einsum("abc,cd->abd", self.input_tmp, self.U1)
            self.inputs_sq1 = tf.squeeze(self.inputs, axis=2)
            self.inputs_sq2 = tf.squeeze(self.inputs_sq1, axis=1)
            self.logits = tf.nn.sigmoid(self.inputs_sq2, name="predict")
        return self.logits

    def loss(self):
        self.loss = tf.losses.log_loss(labels=self.input_y, predictions=self.logits, weights=self.weights,
                                       reduction="weighted_mean")
        self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 0.001
        return self.loss + self.l2_losses

    def instantiate_weights(self):
        with tf.name_scope("Embedding"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer, trainable=True)
