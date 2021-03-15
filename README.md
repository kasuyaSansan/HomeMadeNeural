# HomeMadeNeural

## DeepLearningをC#で作っていくプロジェクト

## お試しし方

### ちょっと速い&高度なコンボリューショナルネット(VGGなど）
+ プロジェクトをクローンし、VisualStudio2019で開く
+ FastConvNetプロジェクトのProgram.csを起動するとサンプルのVGG(コンボリューション3回(ReLUによるアクティベーション含む）、全結合層2回、SoftMax層1回のシンプルなもの)を動作させることができます。
+ 2時間ぐらいで学習が終わります


### コンボリューショナルネット
+ プロジェクトをクローンし、VisualStudio2019で開く
+ ConvNetプロジェクトのProgram.csを起動するとサンプル(5×5のコンボリューション層と3×3のマックスプーリング層×2)を動作させることができます。
+ 1時間ぐらいで学習が終わります

### パーセプトロン
+ プロジェクトをクローンし、VisualStudio2019で開く
+ PerceptronプロジェクトのProgram.csを起動するとサンプル(3層で隠れ層ニューロン数100）を動作させることができます。
+ 1時間ぐらいで学習が終わります


### データについて
- 人文学オープンデータ共同利用センターのKuzushiji-MNISTというデータを北本様の許可をメールでいただき、同梱させていただきました。
- なお、同梱させていただいたデータは最新でない可能性があるため、研究などで理由する場合は本家からダウンロードしていただければと思います。
- Kuzushiji-MNISTのデータはこちらからダウンロードできます。
- http://codh.rois.ac.jp/kmnist/
- github
-- https://github.com/rois-codh/kmnist



### MNISTを自分でダウンロードする

+ http://yann.lecun.com/exdb/mnist/
上記のページから4つのMNISTのgzファイルをダウンロードしてきて、どこかに保存し（デフォルトはc:/data/mnist/の下）
+ pathを保存した場所のものに置き換える
var data = MnistLoader.ReadData(2000, "c:/data/mnist/train-images-idx3-ubyte.gz", "c:/data/mnist/train-labels-idx1-ubyte.gz");
+ PerceptronプロジェクトのProgram.csを起動すると動作させることができます。
+ 1時間ぐらいで学習が終わります
