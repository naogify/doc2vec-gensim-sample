# doc2vec-gensim-sample

Tutorial for Doc2Vec.

Based on https://qiita.com/naotaka1128/items/2c4551abfd40e43b0146.


## Run Docker

1. build image  
`$ docker build -t word2vec:1.0 .`
2. run container  
`$ docker run -it -p 8888:8888 -v ~/Desktop/word2vec-gensim-sample:/home word2vec:1.0`

More info about docker  
https://github.com/naogify/python3-mecab-neologd-dockerfile/blob/master/README.md

## Getting Started

dockerにログイン後、ディレクトリ移動。
`$ cd home/`

### 分かち書き実行
`$ python3 wakati.py`  
`kokoro.txt` に対して分かち書きを実行。`kokoro_wakati.txt` を生成。

### モデル生成
`$ python3 make_model.py`

### 結果確認
`$ python3 result.py`
