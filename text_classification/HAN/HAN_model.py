import tensorflow as tf
class HierarchicalAttention:
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length, num_sentences,
                 vocab_size, embed_size,
                 hidden_size, is_training, need_sentence_level_attention_encoder_flag=True, multi_label_flag=False,
                 initializer=tf.random_normal_initializer(stddev=0.1), clip_gradients=5.0):  # 0.01
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_sentences = num_sentences
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5)
        self.initializer = initializer
        self.multi_label_flag = multi_label_flag
        self.hidden_size = hidden_size
        self.need_sentence_level_attention_encoder_flag = need_sentence_level_attention_encoder_flag
        self.clip_gradients = clip_gradients

        self.input_x=tf.placeholder(tf.int32,[None,self.sequence_length],name="input_x")
        self.input_y=tf.placeholder(tf.int32,[None,self.num_classes],name="input_y")
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.global_step=tf.Variable(0,trainable=False,name="Global_step")
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps,self.decay_rate=decay_steps,decay_rate

        self.instantiate_weights()
        self.logits=self.inference()

        if not is_training:
            return
        self.loss_val=self.loss()
        self.train_op=self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
        correct_prediction = tf.equal(self.predictions, tf.arg_max(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        self.learning_rate=learning_rate
        train_op=tf.contrib.layers.optimize_loss(self.loss_val,global_step=self.global_step,learning_rate=self.learning_rate,optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    def loss(self,l2_lambda=0.0001):
        with tf.name_scope("loss"):
            l2_losses=tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name])*l2_lambda
            losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            loss=l2_losses+tf.reduce_mean(losses)
        return loss

    def inference(self):
        """main computation graph"""
        input_x = tf.split(self.input_x, self.num_sentences,
                           axis=1)  # a list. length:num_sentences.each element is:[None,self.sequence_length/num_sentences]
        input_x = tf.stack(input_x, axis=1)  # shape:[None,self.num_sentences,self.sequence_length/num_sentences]
        embeded_words=tf.nn.embedding_lookup(self.Embedding,input_x) #[None,num_sentences,sentence_length,embed_size]
        word_list=tf.split(embeded_words,self.num_sentences,axis=1)
        embeded_words=[tf.squeeze(x,axis=1) for x in word_list] #a list. length is num_sentences, each element is [None,sentence_length,embed_size]
        word_attention_list=[]
        for i in range(self.num_sentences):
            sentence=embeded_words[i] #[None,sequence_length,embed_size]
            word_encoded=self.bi_lstm(sentence,"word_level",self.hidden_size) #[None,sequence_length,hidden_size*2]

        #     word attention
            word_attention=self.attention(word_encoded,"word_level") #[None,hide_size*2]
            word_attention_list.append(word_attention)
        sentence_encoder_input=tf.stack(word_attention_list,axis=1)#[None,num_sentences,hide_size*2]

        # sentece encoder
        sentence_encoded=self.bi_lstm(sentence_encoder_input,"sentence_level",self.hidden_size*2) #[None,num_sentence,hide_size*2]

        # sentence attention
        document_representation=self.attention(sentence_encoded,"sentence_level") #[None,hide_size*4]
        with tf.name_scope("drop_out"):
            h=tf.nn.dropout(document_representation,keep_prob=self.dropout_keep_prob)#the same as before
        with tf.name_scope("output"):
            logits=tf.layers.dense(h,self.num_classes,use_bias=True)#[None,num_classes]
        return logits





    def bi_lstm(self,input_sequence,level,hidden_size):
        with tf.variable_scope("bi_lstm"+str(level)):
            lstm_fw_cell=tf.contrib.rnn.BasicLSTMCell(hidden_size)
            lstm_bw_cell=tf.contrib.rnn.BasicLSTMCell(hidden_size)
            outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,input_sequence,dtype=tf.float32)  #[None,sequence_length,hide_size*2]
        encoded_outputs=tf.concat(outputs,axis=2)
        return  encoded_outputs

    def attention(self,input_sequences,level):
        num_units=input_sequences.get_shape().as_list()[-1]  #[None,sequence_length,num_units]
        with tf.variable_scope("attention_"+str(level)):
            u=tf.layers.dense(input_sequences,num_units,activation=tf.nn.tanh,use_bias=True)#[None,sequence_length,num_units]
            v_attention=tf.get_variable("u_attention"+level,shape=[num_units],initializer=self.initializer)
            score=tf.multiply(u,v_attention) #[None,seq_length,num_units]
            attention_logits=tf.reduce_sum(score,axis=2)#[None,seq_length]
            attention_logits_max=tf.reduce_max(attention_logits,axis=1,keepdims=True)#[None,1]
            attention=tf.nn.softmax(attention_logits-attention_logits_max) #[None,seq_length]
            attention_extend=tf.expand_dims(attention,axis=2) #[None,seq_length,1]
            sentence_representation=tf.multiply(attention_extend,input_sequences) #[None,seq_length,num_units]

            sentence_representation=tf.reduce_sum(sentence_representation,axis=1) #[None,num_units]

            return sentence_representation



    def instantiate_weights(self):
            with tf.name_scope("Embedding"):
                self.Embedding=tf.get_variable("Embedding",shape=[self.vocab_size,self.embed_size],initializer=self.initializer)