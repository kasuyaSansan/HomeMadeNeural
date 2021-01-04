# HomeMadeNeural

## DeepLearningをC#で作っていくプロジェクト

## お試しし方
+ プロジェクトをクローンし、VisualStudio2019で開く
+ http://yann.lecun.com/exdb/mnist/
上記のページから4つのMNISTのgzファイルをダウンロードしてきて、どこかに保存し（デフォルトはc:/data/mnist/の下）
+ pathを保存した場所のものに置き換える
var data = MnistLoader.ReadData(2000, "c:/data/mnist/train-images-idx3-ubyte.gz", "c:/data/mnist/train-labels-idx1-ubyte.gz");
+ PerceptronプロジェクトのProgram.csを起動すると動作させることができます。
+ 1時間ぐらいで学習が終わります
