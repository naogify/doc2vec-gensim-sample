# ライブラリ読み込み
from gensim import models

# 学習後はモデルをファイルからロード可能
model = models.Doc2Vec.load('./data/doc2vec.model')

# 順番が変わってしまうことがあるので会社リストは学習後に再呼び出し
companies = model.docvecs.offset2doctag

model.most_similar(positive=['良い点'.encode('utf-8')])