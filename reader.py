""" Code from https://github.com/flrngel/TagSpace-tensorflow/blob/master/reader.py
"""

import csv
import numpy as np

class VocabDict(object):
  def __init__(self):
    self.dict = {'<unk>': 0}#初始化词汇表元组

  def fit(self, word):#如果词汇不在词汇表中，将该词对应序号置为词汇表最后一位
    if word not in self.dict:
      self.dict[word] = len(self.dict)

  def size(self):#返回大小
    return len(self.dict)

  def transform(self, word):#如果词汇在词汇表中返回词汇表序号，否则返回0
    if word in self.dict:
      return self.dict[word]
    return 0

  def fit_and_transform(self, word):#将词汇放入词汇表中，并且返回词汇表序号
    self.fit(word)
    return self.transform(word)

def to_categorical(y, target_dict, mode_transform=False):
  result = []
  if mode_transform == False:#若mode_transform=false说明在构建标签集的one-hot向量，否则则是构建词汇表的
    l = len(np.unique(y)) + 1#对标签集合去重排序
  else:
    l = target_dict.size()

  for i, d in enumerate(y):
    tmp = [0.] * l
    for _i, _d in enumerate(d):
      if mode_transform == False:
        tmp[target_dict.fit_and_transform(_d)] = 1.
      else:
        tmp[target_dict.transform(_d)] = 1.
    result.append(tmp)
  return result#返回的是一个矩阵，矩阵由只有0或1组成的向量组成，向量中1的位置指示了该词汇在词汇表中的位置（或者标签集）

def load_csv(filepath, target_columns=-1, columns_to_ignore=None,
    has_header=False, n_classes=None, target_dict=None, mode_transform=False):

  if isinstance(target_columns, list) and len(target_columns) < 1:
    raise Exception('target_columns must be list with one value at least')

  from tensorflow.python.platform import gfile
  with gfile.Open(filepath) as csv_file:
    data_file = csv.reader(csv_file)
    if not columns_to_ignore:
      columns_to_ignore = []
    if has_header:
      header = next(data_file)

    data, target = [], []
    for i, d in enumerate(data_file):
      data.append([_d for _i, _d in enumerate(d) if _i not in target_columns and _i not in columns_to_ignore])
      target.append([_d+str(_i) for _i, _d in enumerate(d) if _i in target_columns])#为什么要加上列号呢？？？

    if target_dict is None:
      target_dict = VocabDict()
    target = to_categorical(target, target_dict=target_dict, mode_transform=mode_transform)
    return data, target#data是api描述矩阵，target是标签0-1向量矩阵
