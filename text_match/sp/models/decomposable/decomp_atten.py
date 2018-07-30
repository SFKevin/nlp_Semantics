import tensorflow as tf


class depos_atten:
    def __init__(self, learning_rate, batch_size, decay_steps, decay_rate, sequence_length, num_sentences,
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
        self.num_sentences = num_sentences
        self.input_x1 = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x2")

        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.weights1 = tf.placeholder(tf.float32, [None], name="weights_place")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_step")
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        self.instantiate_weights()
        self.logits = self.inference()
        self.loss_val = self.loss()

    def inference(self):
        self.input_x1_embed = tf.nn.embedding_lookup(self.Embedding, self.input_x1, name="embed1")
        self.input_x2_embed = tf.nn.embedding_lookup(self.Embedding, self.input_x2, name="embed2")

        with tf.variable_scope("forward_1") as scope:
            self.f1a = tf.layers.dense(self.input_x1_embed, 300, activation=tf.nn.relu, name="F1")
            scope.reuse_variables()
            self.f1b = tf.layers.dense(self.input_x2_embed, 300, activation=tf.nn.relu, name="F1")
        with tf.variable_scope("forward_2") as scope:
            self.fa = tf.layers.dense(self.f1a, 300, activation=tf.nn.relu, name="F2")
            scope.reuse_variables()
            self.fb = tf.layers.dense(self.f1b, 300, activation=tf.nn.relu, name="F2")

        self.weights = tf.matmul(self.fa, self.fb, transpose_b=True, name="weights_mat")
        self.attention_soft1 = tf.nn.softmax(self.weights, name="soft1")
        self.weights_trop = tf.transpose(self.weights, [0, 2, 1], name="weights_trop")
        self.attention_soft2 = tf.nn.softmax(self.weights_trop, name="soft2")
        self.beta = tf.matmul(self.attention_soft1, self.input_x2_embed)
        self.alpha = tf.matmul(self.attention_soft2, self.input_x1_embed)

        self.a_beta = tf.concat([self.input_x1_embed, self.beta], axis=1)
        self.b_alpha = tf.concat([self.input_x2_embed, self.alpha], axis=1)

        with tf.variable_scope("forward_3") as scope:
            self.vi_1 = tf.layers.dense(self.a_beta, 300, activation=tf.nn.relu, name="G1")
            scope.reuse_variables()
            self.vj_1 = tf.layers.dense(self.b_alpha, 300, activation=tf.nn.relu, name="G1")
        with tf.variable_scope("forward_4") as scope:
            self.vi = tf.layers.dense(self.vi_1, 300, activation=tf.nn.relu, name="G2")
            scope.reuse_variables()
            self.vj = tf.layers.dense(self.vj_1, 300, activation=tf.nn.relu, name="G2")

        self.v1 = tf.reduce_sum(self.vi, axis=1)
        self.v2 = tf.reduce_sum(self.vj, axis=1)

        self.element_wise = tf.multiply(self.v1, self.v2)
        self.manha = tf.reduce_sum(
            tf.abs(tf.subtract(self.v1, self.v2)), axis=1,
            keepdims=True)

        self.norm1 = tf.sqrt(tf.reduce_sum(tf.square(self.v1), axis=1, keep_dims=True))
        self.norm2 = tf.sqrt(tf.reduce_sum(tf.square(self.v2), axis=1, keep_dims=True))
        self.sum12 = tf.reduce_sum(self.element_wise, axis=1, keepdims=True)
        self.cos = tf.divide(self.sum12, tf.multiply(self.norm1, self.norm2))

        self.squre = tf.reduce_sum(
            tf.square(tf.subtract(self.v1, self.v2)), axis=1,
            keepdims=True)

        self.inputs = tf.concat(
            [self.element_wise, self.manha, self.norm1, self.norm2, self.sum12, self.cos, self.squre], axis=1)

        with tf.variable_scope("outputs"):
            self.fc1 = tf.layers.dense(self.inputs, 128, activation=tf.nn.relu)
            self.fc1 = tf.nn.dropout(self.fc1, keep_prob=self.dropout_keep_prob)

            self.fc2 = tf.layers.dense(self.fc1, 32, activation=tf.nn.relu)
            self.fc2 = tf.nn.dropout(self.fc2, keep_prob=self.dropout_keep_prob)

            self.fc3 = tf.layers.dense(self.fc2, 1, activation=tf.nn.sigmoid)

            self.logits = tf.squeeze(self.fc3, axis=1, name="predict")
        return self.logits

    def loss(self):
        self.loss = tf.losses.log_loss(labels=self.input_y, predictions=self.logits,
                                       reduction="weighted_mean")
        self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "outputs" in v.name]) * 0.001
        return self.loss + self.l2_losses

    def instantiate_weights(self):
        with tf.name_scope("Embedding"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer, trainable=False)
