# -*- coding: utf-8 -*-

import tensorflow as tf
import tflearn
import numpy as np
import re
from model import SelfAttentive
from sklearn.utils import shuffle
from reader import load_csv, VocabDict
import gensim
from bert_serving.client import BertClient


config = tf.ConfigProto(allow_soft_placement=True)
#最多占gpu资源的70%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
#开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True

'''
parse
'''

tf.app.flags.DEFINE_integer('num_epochs', 15, 'number of epochs to train')
tf.app.flags.DEFINE_integer('batch_size', 20, 'batch size to train in one step')
tf.app.flags.DEFINE_integer('labels', 21, 'number of label classes')
tf.app.flags.DEFINE_integer('word_pad_length', 40, 'word pad length for training')
tf.app.flags.DEFINE_integer('decay_step', 500, 'decay steps')
tf.app.flags.DEFINE_float('learn_rate', 1e-2, 'learn rate for training optimization')
tf.app.flags.DEFINE_boolean('shuffle', True, 'shuffle data FLAG')
tf.app.flags.DEFINE_boolean('train', True, 'train mode FLAG')
tf.app.flags.DEFINE_boolean('visualize', True, 'visualize FLAG')
tf.app.flags.DEFINE_boolean('penalization', False, 'penalization FLAG')

FLAGS = tf.app.flags.FLAGS

num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
tag_size = FLAGS.labels
word_pad_length = FLAGS.word_pad_length
lr = FLAGS.learn_rate

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)
def token_parse(iterator):
  for value in iterator:
    return TOKENIZER_RE.findall(value)

tokenizer = tflearn.data_utils.VocabularyProcessor(word_pad_length, tokenizer_fn=lambda tokens: [token_parse(x) for x in tokens])
label_dict = VocabDict()

def string_parser(arr, fit):
  if fit == False:
    return list(tokenizer.transform(arr))
  else:
    return list(tokenizer.fit_transform(arr))

model = SelfAttentive()
with tf.Session(config=config) as sess:
  # build graph
  model.build_graph(n=word_pad_length)
  # Downstream Application
  with tf.variable_scope('DownstreamApplication'):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learn_rate = tf.train.exponential_decay(lr, global_step, FLAGS.decay_step, 0.95, staircase=True)
    labels = tf.placeholder('float32', shape=[None, tag_size])
    net = tflearn.fully_connected(model.M, 2000, activation='relu')
    logits = tflearn.fully_connected(net, tag_size, activation=None)
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1)
    if FLAGS.penalization == True:  #看不懂
      p_coef = 0.004
      p_loss = p_coef * model.P
      loss = loss + p_loss
      p_loss = tf.reduce_mean(p_loss)
    loss = tf.reduce_mean(loss)
    params = tf.trainable_variables()
    #clipped_gradients = [tf.clip_by_value(x, -0.5, 0.5) for x in gradients]
    optimizer = tf.train.AdamOptimizer(learn_rate)
    grad_and_vars = tf.gradients(loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(grad_and_vars, 0.5)
    opt = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

  # Start Training
  sess.run(tf.global_variables_initializer())
#这里向下都是为了制作句子矩阵
  words, tags = load_csv('./data/train.csv', target_columns=[0], target_dict=label_dict, columns_to_ignore=[1])
  #words = string_parser(words, fit=True)#将每句话的word转化成对应词汇表的序号
  if FLAGS.shuffle == True:
    word_input, tags = shuffle(words, tags)

  # word2vec作为词向量，需要用word2vec.py预训练好word2vec.txt
  '''
  w2v_model = gensim.models.KeyedVectors.load_word2vec_format('./data/pw_20_csv/word2vec.txt', binary=False)
  embedding_matrix = np.zeros((len(tokenizer.vocabulary_._mapping) + 1, 100))
  for word, i in tokenizer.vocabulary_._mapping.items():
    try:
      embedding_matrix[i] = w2v_model.wv[word]#建立对应单词的向量矩阵
    except:
      # words not found in embedding index will be all-zeros.
      # print(str(i)+': '+word+' not in w2v_model.')
      pass

  word_input = tflearn.data_utils.pad_sequences(words, maxlen=word_pad_length)#这里是为了给每句话补齐
  '''
  total = len(word_input)
  step_print = int((total/batch_size) / 12)
  bc = BertClient()#bert-as-servvice的编码器

  if FLAGS.train == True:
    print('start training')
    for epoch_num in range(num_epochs):  
      epoch_loss = 0
      step_loss = 0
      for i in range(int(total/batch_size)):
        #batch_input, batch_tags = (word_input[i*batch_size:(i+1)*batch_size], tags[i*batch_size:(i+1)*batch_size])
        words_batch_tmp = word_input[i*batch_size:(i+1)*batch_size]
        batch_tmp = []
        for k in range(len(words_batch_tmp)):
          batch_tmp.append(''.join(words_batch_tmp[k]))
        batch_input = bc.encode(batch_tmp)
        batch_tags = tags[i*batch_size:(i+1)*batch_size]
        train_ops = [opt, loss, learn_rate, global_step]
        if FLAGS.penalization == True:
          train_ops += [p_loss]
        result = sess.run(train_ops, feed_dict={model.input_embed: batch_input, labels: batch_tags})
        step_loss += result[1]
        epoch_loss += result[1]
        if i % step_print == (step_print-step_print):
          if FLAGS.penalization == True:
            print(f'step_log: (epoch: {epoch_num}, step: {i}, global_step: {result[3]}, learn_rate: {result[2]}), Loss: {step_loss/step_print}, Penalization: {result[4]})')
          else:
            print(f'step_log: (epoch: {epoch_num}, step: {i}, global_step: {result[3]}, learn_rate: {result[2]}), Loss: {step_loss/step_print})')
          #print(f'{result[4]}')
          step_loss = 0
      print('***')
      print(f'epoch {epoch_num}: (global_step: {result[3]}), Average Loss: {epoch_loss/(total/batch_size)})')
      print('***\n')
    saver = tf.train.Saver()
    saver.save(sess, './output/model.ckpt')
  else:
    saver = tf.train.Saver()
    saver.restore(sess, './output/model.ckpt')
  
  word_input, tags = load_csv('./data/test.csv', target_columns=[0], target_dict=label_dict, columns_to_ignore=[1])
  #words_with_index = string_parser(words, fit=True)
  #word_input = tflearn.data_utils.pad_sequences(words_with_index, maxlen=word_pad_length)
  total = len(word_input)
  rs = 0.

  if FLAGS.visualize == True:
    f = open('visualize_1.html', 'w', encoding="utf-8")
    f.write('<html style="margin:0;padding:0;"><body style="margin:0;padding:0;">\n')

  for i in range(int(total/batch_size)):
    #batch_input, batch_tags = (word_input[i*batch_size:(i+1)*batch_size], tags[i*batch_size:(i+1)*batch_size])
    words_batch_tmp = word_input[i*batch_size:(i+1)*batch_size]
    batch_tmp = []
    for k in range(len(words_batch_tmp)):
      batch_tmp.append(''.join(words_batch_tmp[k]))
    batch_input = bc.encode(batch_tmp)
    batch_tags = tags[i*batch_size:(i+1)*batch_size]
    result = sess.run([logits, model.A], feed_dict={model.input_embed: batch_input, labels: batch_tags})
    arr = result[0]
    for j in range(len(batch_tags)):
      rs+=np.sum(np.argmax(arr[j]) == np.argmax(batch_tags[j]))

    if FLAGS.visualize == True:
      f.write('<div style="margin:25px;">\n')
      for k in range(len(result[1][0])):
        f.write('<p style="margin:10px;">\n')
        ww = TOKENIZER_RE.findall(word_input[i*batch_size][0])
        for j in range(word_pad_length):
          alpha = "{:.2f}".format(result[1][0][k][j])
          if len(ww) <= j:
            w = "___"
          else:
            w = ww[j]
          f.write(f'\t<span style="margin-left:3px;background-color:rgba(255,0,0,{alpha})">{w}</span>\n')
        f.write('</p>\n')
      f.write('</div>\n')

  if FLAGS.visualize == True:
    f.write('</body></html>')
    f.close()
  print(f'Test accuracy: {rs/total}')

  sess.close()
