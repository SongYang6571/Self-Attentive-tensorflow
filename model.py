# -*- coding: utf-8 -*-

import tensorflow as tf
import gensim

class SelfAttentive(object):
  '''
  Tensorflow implementation of 'A Structured Self Attentive Sentence Embedding'
  (https://arxiv.org/pdf/1703.03130.pdf)
  '''
  def build_graph(self, n=40, d=100, u=128, d_a=350, r=10, reuse=False):
    with tf.variable_scope('SelfAttentive', reuse=reuse):
      # Hyperparmeters from paper
      self.n = n        # sentence_len
      self.d = d        # embedding维度
      self.d_a = d_a    # 可训练变量W_s1和W_s2的维度超参数
      self.u = u        # LSTM的隐藏单元数
      self.r = r        # multiple hops of attention

      initializer = tf.contrib.layers.xavier_initializer()

    #   embedding = tf.get_variable('embedding', shape=[100000, self.d],
    #       initializer=initializer)
      #self.embedding = tf.placeholder(tf.float32, shape=[None, self.d])#全体词向量
      self.input_embed = tf.placeholder(tf.float32, shape=[None, self.n, 768])    # batch个句子,每个句子中词向量768维

      # Declare trainable variables
      # shape(W_s1) = d_a * 2u
      self.W_s1 = tf.get_variable('W_s1', shape=[self.d_a, 2*self.u],
          initializer=initializer)
      # shape(W_s2) = r * d_a
      self.W_s2 = tf.get_variable('W_s2', shape=[self.r, self.d_a],
          initializer=initializer)

      # BiRNN
      self.batch_size = batch_size = tf.shape(self.input_embed)[0]

      cell_fw = tf.contrib.rnn.LSTMCell(u)
      cell_bw = tf.contrib.rnn.LSTMCell(u)
      input_embed = self.input_embed
      H, _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw,
          cell_bw,
          input_embed,
          dtype=tf.float32)
      H = tf.concat([H[0], H[1]], axis=2)   # [batch_size, n, 2u]

      self.A = A = tf.nn.softmax(
          tf.map_fn(
            lambda x: tf.matmul(self.W_s2, x), 
            tf.tanh(
              tf.map_fn(
                lambda x: tf.matmul(self.W_s1, tf.transpose(x)),
                H))))   # [batch_size, r, n]

      self.M = tf.matmul(A, H)  # 句子用矩阵表示 [batch_size, r, 2u]

      A_T = tf.transpose(A, perm=[0, 2, 1])             # [batch_size, n, r]
      tile_eye = tf.tile(tf.eye(r), [batch_size, 1])    # [batch_size, r]
      tile_eye = tf.reshape(tile_eye, [-1, r, r])       # [batch_size, r, r]
      AA_T = tf.matmul(A, A_T) - tile_eye               # [batch_size, r, r]
      self.P = tf.square(tf.norm(AA_T, axis=[-2, -1], ord='fro'))   # [batch_size]

  def trainable_vars(self):
    return [var for var in
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SelfAttentive')]
