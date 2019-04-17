import gensim
import codecs

sentences = gensim.models.word2vec.PathLineSentences('./data/pw_20_csv/data_20_shuffle_stemmer.txt')  # 按行读入
model = gensim.models.Word2Vec(sentences, min_count=0,size=32)
# bigram_transformer = gensim.models.Phrases(sentences)   # 自动检测短语的词向量
# model = gensim.models.Word2Vec(bigram_transformer[sentences], min_count=0)

# 两种保存模型的方法
# model.save('./data/pw_20_csv/word2vec')

#model.wv.save_word2vec_format('./data/pw_20_csv/word2vec.txt', binary=False)
model.wv.save_word2vec_format('./data/pw_20_csv/word2vec_1.txt', binary=False)

# 两种load方法对应不同的保存方法
# model2 = Word2Vec.load('./data/pw_20_csv/word2vec')
model2 = gensim.models.KeyedVectors.load_word2vec_format('./data/pw_20_csv/word2vec_1.txt', binary=False)

print(model2.most_similar('api'))
print(model2.wv['api'])