import sys
sys.path.append("./")

# クチコミデータの読み込み
import data
rows = data.rows

# 参考記事のstem関数で語幹を抽出
from utils import stems  # 参考記事の実装ほぼそのまま
companies = [row[0] for row in rows]
docs = [stems(row[1]) for row in rows]

# デバッグ用
# print(docs)

# ライブラリ読み込み
from gensim import models

class LabeledListSentence(object):
    def __init__(self, words_list, labels):
        self.words_list = words_list
        self.labels = labels

    def __iter__(self):
        for i, words in enumerate(self.words_list):
            yield models.doc2vec.LabeledSentence(words, ['%s' % self.labels[i]])

# gensim にクチコミを登録
# クチコミに会社名を付与するため、参考記事で実装されていた拡張クラスを使っています
sentences = LabeledListSentence(docs, companies)

# doc2vec の学習条件設定
# alpha: 学習率 / min_count: X回未満しか出てこない単語は無視
# size: ベクトルの次元数 / iter: 反復回数 / workers: 並列実行数
model = models.Doc2Vec(alpha=0.025, min_count=5,
                       vector_size=100, epochs=20, workers=4)

# doc2vec の学習前準備(単語リスト構築)
model.build_vocab(sentences)

# Wikipedia から学習させた単語ベクトルを無理やり適用して利用することも出来ます
# model.intersect_word2vec_format('./data/wiki/wiki2vec.bin', binary=True)


# 学習実行
model.train(
    sentences,
    total_examples = len(rows),
    epochs = model.epochs,
)

# セーブ
model.save('./data/doc2vec.model')

# 学習後はモデルをファイルからロード可能
# model = models.Doc2Vec.load('./data/doc2vec.model')

# 順番が変わってしまうことがあるので会社リストは学習後に再呼び出し
companies = model.docvecs.offset2doctag