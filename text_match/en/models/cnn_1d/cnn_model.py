import tensorflow as tf


class Cnn_1d:
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
        self.input_x1_exp = tf.expand_dims(self.input_x1_embed, -1)
        self.input_x2_exp = tf.expand_dims(self.input_x2_embed, -1)
        with tf.variable_scope("conv1") as scope:
            self.conv1_1 = tf.layers.conv2d(self.input_x1_exp, filters=64, kernel_size=[1, self.embed_size],
                                            strides=[1, 1], padding='SAME', activation=tf.nn.relu,
                                            kernel_initializer=tf.variance_scaling_initializer(), name="conv1")
            self.globa1_1 = tf.keras.layers.GlobalAveragePooling2D()(self.conv1_1)
            scope.reuse_variables()
            self.conv1_2 = tf.layers.conv2d(self.input_x2_exp, filters=64, kernel_size=[1, self.embed_size],
                                            strides=[1, 1], padding='SAME', activation=tf.nn.relu,
                                            kernel_initializer=tf.variance_scaling_initializer(), name="conv1")
            self.globa1_2 = tf.keras.layers.GlobalAveragePooling2D()(self.conv1_2)
        with tf.variable_scope("conv2") as scope:
            self.conv2_1 = tf.layers.conv2d(self.input_x1_exp, filters=64, kernel_size=[2, self.embed_size],
                                            strides=[1, 1], padding='SAME', activation=tf.nn.relu,
                                            kernel_initializer=tf.variance_scaling_initializer(), name="conv2")
            self.globa2_1 = tf.keras.layers.GlobalAveragePooling2D()(self.conv2_1)
            scope.reuse_variables()
            self.conv2_2 = tf.layers.conv2d(self.input_x2_exp, filters=64, kernel_size=[2, self.embed_size],
                                            strides=[1, 1], padding='SAME', activation=tf.nn.relu,
                                            kernel_initializer=tf.variance_scaling_initializer(), name="conv2")
            self.globa2_2 = tf.keras.layers.GlobalAveragePooling2D()(self.conv2_2)
        with tf.variable_scope("conv3") as scope:
            self.conv3_1 = tf.layers.conv2d(self.input_x1_exp, filters=64, kernel_size=[3, self.embed_size],
                                            strides=[1, 1], padding='SAME', activation=tf.nn.relu,
                                            kernel_initializer=tf.variance_scaling_initializer(), name="conv3")
            self.globa3_1 = tf.keras.layers.GlobalAveragePooling2D()(self.conv3_1)
            scope.reuse_variables()
            self.conv3_2 = tf.layers.conv2d(self.input_x2_exp, filters=64, kernel_size=[3, self.embed_size],
                                            strides=[1, 1], padding='SAME', activation=tf.nn.relu,
                                            kernel_initializer=tf.variance_scaling_initializer(), name="conv3")
            self.globa3_2 = tf.keras.layers.GlobalAveragePooling2D()(self.conv3_2)

        # with tf.variable_scope("conv4") as scope:
        #     self.conv4_1 = tf.layers.conv2d(self.input_x1_exp, filters=128, kernel_size=[4, self.embed_size],
        #                                     strides=[1, 1], padding='SAME', activation=tf.nn.relu,
        #                                     kernel_initializer=tf.variance_scaling_initializer(), name="conv4")
        #     self.globa4_1 = tf.keras.layers.GlobalAveragePooling2D()(self.conv4_1)
        #     scope.reuse_variables()
        #     self.conv4_2 = tf.layers.conv2d(self.input_x2_exp, filters=128, kernel_size=[4, self.embed_size],
        #                                     strides=[1, 1], padding='SAME', activation=tf.nn.relu,
        #                                     kernel_initializer=tf.variance_scaling_initializer(), name="conv4")
        #     self.globa4_2 = tf.keras.layers.GlobalAveragePooling2D()(self.conv4_2)
        # with tf.variable_scope("conv5") as scope:
        #     self.conv5_1 = tf.layers.conv2d(self.input_x1_exp, filters=64, kernel_size=[5, self.embed_size],
        #                                     strides=[1, 1], padding='SAME', activation=tf.nn.relu,
        #                                     kernel_initializer=tf.variance_scaling_initializer(), name="conv5")
        #     self.globa5_1 = tf.keras.layers.GlobalAveragePooling2D()(self.conv5_1)
        #     scope.reuse_variables()
        #     self.conv5_2 = tf.layers.conv2d(self.input_x2_exp, filters=64, kernel_size=[5, self.embed_size],
        #                                     strides=[1, 1], padding='SAME', activation=tf.nn.relu,
        #                                     kernel_initializer=tf.variance_scaling_initializer(), name="conv5")
        #     self.globa5_2 = tf.keras.layers.GlobalAveragePooling2D()(self.conv5_2)
        # with tf.variable_scope("conv6") as scope:
        #     self.conv6_1 = tf.layers.conv2d(self.input_x1_exp, filters=64, kernel_size=[6, self.embed_size],
        #                                     strides=[1, 1], padding='SAME', activation=tf.nn.relu,
        #                                     kernel_initializer=tf.variance_scaling_initializer(), name="conv6")
        #     self.globa6_1 = tf.keras.layers.GlobalAveragePooling2D()(self.conv6_1)
        #     scope.reuse_variables()
        #     self.conv6_2 = tf.layers.conv2d(self.input_x2_exp, filters=64, kernel_size=[6, self.embed_size],
        #                                     strides=[1, 1], padding='SAME', activation=tf.nn.relu,
        #                                     kernel_initializer=tf.variance_scaling_initializer(), name="conv6")
        #     self.globa6_2 = tf.keras.layers.GlobalAveragePooling2D()(self.conv6_2)
        self.merge1 = tf.concat(
            [self.globa1_1, self.globa2_1, self.globa3_1], axis=1)
        self.merge2 = tf.concat(
            [self.globa1_2, self.globa2_2, self.globa3_2], axis=1)

        self.diff = tf.abs(tf.subtract(self.merge1, self.merge2))
        self.multi = tf.multiply(self.merge1, self.merge2)
        self.fc = tf.concat([self.diff, self.multi], axis=1)
        self.fc_drop = tf.nn.dropout(self.fc, keep_prob=self.dropout_keep_prob)
        with tf.variable_scope("outputs"):
            self.fc1 = tf.layers.dense(self.fc_drop, 256, activation=tf.nn.relu,
                                       kernel_initializer=tf.variance_scaling_initializer())
            self.fc1_drop = tf.nn.dropout(self.fc1, keep_prob=self.dropout_keep_prob)

            self.fc2 = tf.layers.dense(self.fc1_drop, 128, activation=tf.nn.relu,
                                       kernel_initializer=tf.variance_scaling_initializer())
            self.fc2_drop = tf.nn.dropout(self.fc2, keep_prob=self.dropout_keep_prob)

            self.fc3 = tf.layers.dense(self.fc2_drop, 1, activation=tf.nn.sigmoid,
                                       kernel_initializer=tf.variance_scaling_initializer())

            self.logits = tf.squeeze(self.fc3, axis=1, name="predict")
            return self.logits

    def loss(self):
        self.loss = tf.losses.log_loss(labels=self.input_y, predictions=self.logits,
                                       reduction="weighted_mean")
        self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 0.0001
        return self.loss + self.l2_losses

    def instantiate_weights(self):
        with tf.name_scope("Embedding"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer, trainable=True)
