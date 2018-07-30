import tensorflow as tf


class HierarchicalAttention:
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
        self.weights = tf.placeholder(tf.float32, [None], name="weights")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_step")
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        self.instantiate_weights()
        self.logits = self.inference()
        self.loss_val = self.loss()

    def inference(self):
        """main computation graph"""
        input_x1 = tf.split(self.input_x1, self.num_sentences,
                            axis=1)  # a list. length:num_sentences.each element is:[None,self.sequence_length/num_sentences]
        input_x1 = tf.stack(input_x1, axis=1)  # shape:[None,self.num_sentences,self.sequence_length/num_sentences]
        embeded_words1 = tf.nn.embedding_lookup(self.Embedding,
                                                input_x1)  # [None,num_sentences,sentence_length,embed_size]
        word_list1 = tf.split(embeded_words1, self.num_sentences, axis=1)
        embeded_words1 = [tf.squeeze(x, axis=1) for x in
                          word_list1]  # a list. length is num_sentences, each element is [None,sentence_length,embed_size]
        word_attention_list1 = []
        with tf.variable_scope("biLSTM") as scope:
            for i in range(self.num_sentences):
                sentence1 = embeded_words1[i]  # [None,sequence_length,embed_size]
                word_encoded1 = self.bi_lstm(sentence1, "word_level1",
                                             self.hidden_size)  # [None,sequence_length,hidden_size*2]

                word_attention1 = self.attention(word_encoded1, "word_level1")  # [None,hide_size*2]
                word_attention_list1.append(word_attention1)
            sentence_encoder_input1 = tf.stack(word_attention_list1, axis=1)  # [None,num_sentences,hide_size*2]
            scope.reuse_variables()
            input_x2 = tf.split(self.input_x2, self.num_sentences,
                                axis=1)  # a list. length:num_sentences.each element is:[None,self.sequence_length/num_sentences]
            input_x2 = tf.stack(input_x2, axis=1)  # shape:[None,self.num_sentences,self.sequence_length/num_sentences]
            embeded_words2 = tf.nn.embedding_lookup(self.Embedding,
                                                    input_x2)  # [None,num_sentences,sentence_length,embed_size]
            word_list2 = tf.split(embeded_words2, self.num_sentences, axis=1)
            embeded_words2 = [tf.squeeze(x, axis=1) for x in
                              word_list2]  # a list. length is num_sentences, each element is [None,sentence_length,embed_size]
            word_attention_list2 = []
            for i in range(self.num_sentences):
                sentence2 = embeded_words2[i]  # [None,sequence_length,embed_size]
                word_encoded2 = self.bi_lstm(sentence2, "word_level1",
                                             self.hidden_size)  # [None,sequence_length,hidden_size*2]

                #     word attention
                word_attention2 = self.attention(word_encoded2, "word_level1")  # [None,hide_size*2]
                word_attention_list2.append(word_attention2)
            sentence_encoder_input2 = tf.stack(word_attention_list2, axis=1)  # [None,num_sentences,hide_size*2]

        self.sentence_encoder_input1_sq = tf.squeeze(sentence_encoder_input1, axis=1)
        self.sentence_encoder_input2_sq = tf.squeeze(sentence_encoder_input2, axis=1)

        with tf.variable_scope("outputs"):
            self.element_wise = tf.multiply(self.sentence_encoder_input1_sq, self.sentence_encoder_input2_sq)
            self.manha = tf.reduce_sum(
                tf.abs(tf.subtract(self.sentence_encoder_input1_sq, self.sentence_encoder_input1_sq)), axis=1,
                keepdims=True)

            self.norm1 = tf.sqrt(tf.reduce_sum(tf.square(self.sentence_encoder_input1_sq), axis=1, keep_dims=True))
            self.norm2 = tf.sqrt(tf.reduce_sum(tf.square(self.sentence_encoder_input2_sq), axis=1, keep_dims=True))
            self.sum12 = tf.reduce_sum(self.element_wise, axis=1, keepdims=True)
            self.cos = tf.divide(self.sum12, tf.multiply(self.norm1, self.norm2))

            self.squre = tf.reduce_sum(
                tf.square(tf.subtract(self.sentence_encoder_input1_sq, self.sentence_encoder_input1_sq)), axis=1,
                keepdims=True)

            self.inputs = tf.concat(
                [self.element_wise, self.manha, self.cos, self.squre], axis=1)

            self.inputs_bn = tf.layers.batch_normalization(self.inputs, training=self.is_training)
            self.fc1 = tf.layers.dense(self.inputs_bn, 128, activation=tf.nn.relu)
            self.fc1 = tf.nn.dropout(self.fc1, keep_prob=self.dropout_keep_prob)

            self.fc2 = tf.layers.dense(self.fc1, 64, activation=tf.nn.relu)
            self.fc2 = tf.nn.dropout(self.fc2, keep_prob=self.dropout_keep_prob)

            self.fc3 = tf.layers.dense(self.fc2, 1, activation=tf.nn.sigmoid)
            self.logits = tf.squeeze(self.fc3, axis=1,
                                     name="predict")
            # self.logits = tf.nn.softmax(self.fc3, name="predict")
            # self.classes = tf.argmax(input=self.logits, axis=1, name='classes')
            return self.logits

    def loss(self):
        # self.onehot_labels = tf.one_hot(indices=tf.cast(self.input_y, tf.int32), depth=2)
        self.loss = tf.losses.log_loss(labels=self.input_y, predictions=self.logits, weights=self.weights,
                                       reduction="weighted_mean")
        # self.l1_l2_losses = tf.add_n(
        #     [tf.contrib.layers.l2_regularizer(scale=0.02)(v) for v in tf.trainable_variables() if
        #      "outputs" in v.name])
        self.l1_l2_losses = tf.add_n(
            [tf.contrib.layers.l2_regularizer(scale=0.01)(v) for v in tf.trainable_variables()])
        # self.l1_losses = tf.add_n(
        #     [tf.abs(v) for v in tf.trainable_variables() if "outputs" in v.name]) * 0.001
        return self.loss + self.l1_l2_losses

    def bi_lstm(self, input_sequence, level, hidden_size):
        with tf.variable_scope("bi_lstm" + str(level)):
            lstm_fw_cell = tf.contrib.rnn.GRUCell(hidden_size)
            lstm_bw_cell = tf.contrib.rnn.GRUCell(hidden_size)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_sequence,
                                                         dtype=tf.float32)  # [None,sequence_length,hide_size*2]
        encoded_outputs = tf.concat(outputs, axis=2)
        return encoded_outputs

    def attention(self, input_sequences, level):
        num_units = input_sequences.get_shape().as_list()[-1]  # [None,sequence_length,num_units]
        with tf.variable_scope("attention_" + str(level)):
            u = tf.layers.dense(input_sequences, num_units, activation=tf.nn.tanh,
                                use_bias=True)  # [None,sequence_length,num_units]
            v_attention = tf.get_variable("u_attention" + level, shape=[num_units], initializer=self.initializer)
            score = tf.multiply(u, v_attention)  # [None,seq_length,num_units]
            attention_logits = tf.reduce_sum(score, axis=2)  # [None,seq_length]
            attention_logits_max = tf.reduce_max(attention_logits, axis=1, keepdims=True)  # [None,1]
            attention = tf.nn.softmax(attention_logits - attention_logits_max)  # [None,seq_length]
            attention_extend = tf.expand_dims(attention, axis=2)  # [None,seq_length,1]
            sentence_representation = tf.multiply(attention_extend, input_sequences)  # [None,seq_length,num_units]

            sentence_representation = tf.reduce_sum(sentence_representation, axis=1)  # [None,num_units]

            return sentence_representation

    def instantiate_weights(self):
        with tf.name_scope("Embedding"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer, trainable=False)
