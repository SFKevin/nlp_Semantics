import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


class cnn_text:
    def __init__(self, learning_rate, batch_size, decay_steps, decay_rate, sequence_length, num_sentences,
                 hidden_size, vocab_size, embed_size,
                 initializer=tf.random_normal_initializer(stddev=0.1), clip_gradients=5.0):  # 0.01
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
        self.weights = tf.placeholder(tf.float32, [None], name="weights")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_step")
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        self.instantiate_weights()
        self.logits = self.inference()
        self.loss_val = self.loss()

    def loss(self):
        self.loss = tf.losses.log_loss(labels=self.input_y, predictions=self.logits,
                                       reduction="weighted_mean")
        self.l2_losses = tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 0.01
        return self.loss + self.l2_losses

    def inference(self):
        self.input_x1_embed = tf.nn.embedding_lookup(self.Embedding, self.input_x1)
        self.input_x2_embed = tf.nn.embedding_lookup(self.Embedding, self.input_x2)
        # self.input_x2_trop = tf.transpose(self.input_x2_embed, [0, 2, 1])

        # self.image = tf.matmul(self.input_x1_embed, self.input_x2_trop, name="image")
        # self.image_exp = tf.expand_dims(self.image, axis=-1)
        self.dot_wise = tf.matmul(self.input_x1_embed, self.input_x2_embed, transpose_b=True)
        self.norm1 = tf.sqrt(tf.reduce_sum(tf.square(self.input_x1_embed), axis=2, keep_dims=True), name="norm1")
        self.norm2 = tf.sqrt(tf.reduce_sum(tf.square(self.input_x2_embed), axis=2, keep_dims=True), name="norm2")
        self.norm = tf.matmul(self.norm1, self.norm2, transpose_b=True)
        self.cos = tf.div(self.dot_wise, self.norm, name="cos")
        self.image_exp = tf.expand_dims(self.cos, axis=-1)
        with tf.variable_scope("conv1"):
            self.conv1 = tf.layers.conv2d(inputs=self.image_exp, filters=128, kernel_size=3, strides=[1, 1],
                                          padding="SAME"
                                          , use_bias=True,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv")
            self.bn1 = tf.layers.batch_normalization(inputs=self.conv1, training=self.is_training)
            self.activ1 = tf.nn.relu(self.bn1)

            self.pool1 = tf.layers.average_pooling2d(self.activ1, pool_size=2, strides=[2, 2], padding="SAME",
                                                     name="pool")

        with tf.variable_scope("conv2"):
            self.conv2 = tf.layers.conv2d(inputs=self.pool1, filters=128, kernel_size=3, strides=[1, 1], padding="SAME",
                                          use_bias=True,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv")
            self.bn2 = tf.layers.batch_normalization(inputs=self.conv2, training=self.is_training)
            self.activ2 = tf.nn.relu(self.bn2)

            self.pool2 = tf.layers.average_pooling2d(self.activ2, pool_size=2, strides=[2, 2], padding="SAME",
                                                     name="pool")
        with tf.variable_scope("conv3"):
            self.conv3 = tf.layers.conv2d(inputs=self.pool2, filters=128, kernel_size=3, strides=[1, 1], padding="SAME",
                                          use_bias=True,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv")
            self.bn3 = tf.layers.batch_normalization(inputs=self.conv3, training=self.is_training)
            self.activ3 = tf.nn.relu(self.bn3)

            self.pool3 = tf.keras.layers.GlobalAveragePooling2D()(self.activ3)
        # self.fc = tf.layers.flatten(self.pool2, name="flatten")
        with tf.name_scope("outputs"):
            self.fc1 = tf.layers.dense(self.pool3, 512, activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")
            self.drop1 = tf.nn.dropout(self.fc1, keep_prob=self.dropout_keep_prob, name="drop1")

            self.fc2 = tf.layers.dense(self.drop1, 128, activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc2")
            self.drop2 = tf.nn.dropout(self.fc2, keep_prob=self.dropout_keep_prob, name="drop2")

            self.logits = tf.squeeze(tf.layers.dense(self.drop2, 1, activation=tf.nn.sigmoid), axis=1, name="predict")
        return self.logits

    def instantiate_weights(self):
        with tf.name_scope("Embedding"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer, trainable=True)
