import tensorflow as tf
from res_char_cnn import datautils

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


class ResNet():
    def __init__(self, l0, num_class, l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.int32, [None, l0], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_class], name="input_y")
        self.training = tf.placeholder(tf.bool)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)

        with tf.name_scope("embedding"):
            self.W, _ = datautils.onehot_dic_build()
            self.x_image = tf.nn.embedding_lookup(self.W, self.input_x)
            self.x_flat = tf.expand_dims(self.x_image, -1)  # [batch_size,seq_length,embedd_size,1]

        with tf.name_scope("init_conv"):
            pad_total = 3 - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            inputs = tf.pad(self.x_flat, [[0, 0], [pad_beg, pad_end], [0, 0], [0, 0]])
            input_64 = tf.layers.conv2d(inputs, filters=64, kernel_size=[3, 69], strides=[1, 1], padding='VALID',
                                        kernel_initializer=tf.variance_scaling_initializer())  # [batch_size,seq_length,1,64]
        with tf.name_scope("blocks"):
            input_64 = self._building_block(input_64, 64, self.training, "conv_3_64")  # [[batch_size,seq_length,1,64]]
            pool_64 = tf.layers.max_pooling2d(input_64, pool_size=[3, 1], strides=[2, 1], padding='SAME',
                                              name="pool_3_64")
            input_128 = tf.layers.conv2d(pool_64, 128, kernel_size=1, strides=[1, 1], padding='SAME',
                                         name="conv_preject128")

            input_128 = self._building_block(input_128, 128, self.training, "conv_3_128")
            pool_128 = tf.layers.max_pooling2d(input_128, pool_size=[3, 1], strides=[2, 1], padding='SAME',
                                               name="pool_3_128")

            input_256 = tf.layers.conv2d(pool_128, 256, kernel_size=1, strides=[1, 1], padding='SAME',
                                         name="conv_preject256")

            input_256 = self._building_block(input_256, 256, self.training, "conv_3_256")
            pool_256 = tf.layers.max_pooling2d(input_256, pool_size=[3, 1], strides=[2, 1], padding='SAME',
                                               name="pool_3_256")

            input_512 = tf.layers.conv2d(pool_256, 512, kernel_size=1, strides=[1, 1], padding='SAME',
                                         name="conv_preject512")

            input_512 = self._building_block(input_512, 512, self.training, "conv_3_512")
            pool_512 = tf.layers.max_pooling2d(input_512, pool_size=[3, 1], strides=[2, 1], padding='SAME',
                                               name="pool_3_512")
            final = tf.transpose(pool_512, [0, 3, 2, 1])
            pooled = tf.nn.top_k(final, k=8, name='k-maxpooling')
            last = tf.reshape(pooled[0], (-1, 512 * 8))

        with tf.name_scope("fc123"):
            fc1_out, fc1_loss = self.linear(last, 2048, scopes="fc1", stddev=0.1)
            fc1_out=tf.nn.dropout(fc1_out,self.dropout_keep_prob)
            l2_loss += fc1_loss
            fc2_out, fc2_loss = self.linear(tf.nn.relu(fc1_out), 2048, scopes="fc2", stddev=0.1)
            l2_loss += fc2_loss
            fc2_out=tf.nn.dropout(fc2_out,self.dropout_keep_prob)
            fc3_out, fc3_loss = self.linear(tf.nn.relu(fc2_out), num_class, scopes="fc3", stddev=0.1)
            l2_loss += fc3_loss

        with tf.name_scope("loss"):
            self.predictions = tf.argmax(fc3_out, 1, name="predictions")
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=fc3_out, labels=self.input_y)
            self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def _building_block(self, inputs, filters, training, scope):
        with tf.name_scope(scope):
            shortcut = inputs
            inputs = tf.layers.conv2d(inputs, filters, kernel_size=[3, 1], strides=[1, 1], padding='SAME',
                                      kernel_initializer=tf.variance_scaling_initializer())
            bn1 = tf.layers.batch_normalization(inputs=inputs, axis=3, momentum=_BATCH_NORM_DECAY,
                                                epsilon=_BATCH_NORM_EPSILON, center=True,
                                                scale=True, training=training, fused=True)

            inputs = tf.nn.relu(bn1)
            inputs = tf.layers.conv2d(inputs, filters, kernel_size=[3, 1], strides=[1, 1], padding='SAME',
                                      kernel_initializer=tf.variance_scaling_initializer())
            bn2 = tf.layers.batch_normalization(inputs=inputs, axis=3, momentum=_BATCH_NORM_DECAY,
                                                epsilon=_BATCH_NORM_EPSILON, center=True,
                                                scale=True, training=training, fused=True)

            inputs = bn2 + shortcut
            inputs = tf.nn.relu(inputs)
        return inputs

    def linear(self, input, output_dim, scopes, stddev=0.1):
        norm = tf.random_normal_initializer(stddev=stddev)
        const = tf.constant_initializer(0.0)
        with tf.variable_scope(scopes):
            w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
            b = tf.get_variable('b', [output_dim], initializer=const)
            l2_loss = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
        return tf.matmul(input, w) + b, l2_loss
