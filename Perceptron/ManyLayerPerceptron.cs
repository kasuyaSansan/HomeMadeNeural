using System;
using System.Collections.Generic;
using System.Linq;
using NeuralLib;

namespace Perceptron
{
    public class ManyLayerPerceptron
    {
        private double[][] states;//states[n][k] n層目のk番目のニューロンの状態
        private double[][] outputs;//outputs[n][k] n層目のk番目のニューロンの出力
        internal double[][,] weights;//weights[n][k,l] n層目のk番目のニューロンとn+1層目のl番目のニューロンの重み
        private double[][,] deltaWeights;
        private List<TrainingData> sample = new List<TrainingData>();//学習サンプル
        private int[] numNeuron;

        public double e = 0.001;
        private double T = 2;
        private int numLayer;

        private delegate double ActivationFunction(double x, double param);
        private readonly ActivationFunction activatFunc;
        private delegate double DifferentialActivationFunction(double x, double param);
        private readonly DifferentialActivationFunction differentialFunc;

        public int batchSize = 100;

        /// <summary>
        /// コンストラクタ
        /// </summary>
        /// <param name="num1st"></param>
        /// <param name="num2nd"></param>
        /// <param name="num3rd"></param>
        public ManyLayerPerceptron(int[] numNeuron, double learnRate = 0.001, FUNC_TYPE funcType = FUNC_TYPE.Sigmoid, int batchSize = 100)
        {
            this.numNeuron = numNeuron;
            this.numLayer = numNeuron.Length;
            e = learnRate;
            this.batchSize = batchSize;
            if (funcType == FUNC_TYPE.Sigmoid)
            {
                activatFunc = NeuralFunc.Sigmoid;
                differentialFunc = NeuralFunc.Dsigmoid;
            }
            else
            {
                activatFunc = NeuralFunc.ReLU;
                differentialFunc = NeuralFunc.DReLU;
            }
            states = new double[numLayer][];
            outputs = new double[numLayer][];
            weights = new double[numLayer - 1][,];
            deltaWeights = new double[numLayer - 1][,];


            for (var i = 0; i < numLayer; i++)
            {
                states[i] = new double[numNeuron[i]];
                outputs[i] = new double[numNeuron[i]];
                if (i != numLayer - 1)
                {
                    weights[i] = new double[numNeuron[i], numNeuron[i + 1]];
                    deltaWeights[i] = new double[numNeuron[i], numNeuron[i + 1]];
                }
            }

            //重みを乱数で初期化
            var r = new Random();
            for (var n = 0; n < numLayer - 1; n++)
            {
                for (var i = 0; i < numNeuron[n]; i++)
                {
                    for (var j = 0; j < numNeuron[n + 1]; j++)
                    {
                        weights[n][i, j] = NeuralFunc.BoxMuler(0, Math.Sqrt(2.0 / numNeuron[n]));
                    }
                }
            }
        }

        private void ResetDeltaWeight()
        {
            for (var n = 0; n < numLayer - 1; n++)
            {
                for (var i = 0; i < numNeuron[n]; i++)
                {
                    for (var j = 0; j < numNeuron[n + 1]; j++)
                    {
                        deltaWeights[n][i, j] = 0;
                    }
                }
            }
        }

        private void AddDeltaWeight()
        {
            var absMax = 0.0;

            for (var n = 0; n < numLayer - 1; n++)
            {
                for (var i = 0; i < numNeuron[n]; i++)
                {
                    for (var j = 0; j < numNeuron[n + 1]; j++)
                    {
                        absMax = Math.Max(Math.Abs(deltaWeights[n][i, j]), absMax);
                    }
                }
            }

            // ReLUを使うときに発散しにくくするため、絶対値が大きすぎるときは係数を小さくする
            var coef = 1.0;
            var ratio = absMax / Math.Sqrt(2.0 / numNeuron.Max());
            if (ratio > 0.05)
            {
                coef = 0.05 / ratio;
            }

            for (var n = 0; n < numLayer - 1; n++)
            {
                for (var i = 0; i < numNeuron[n]; i++)
                {
                    for (var j = 0; j < numNeuron[n + 1]; j++)
                    {
                        weights[n][i, j] += coef * deltaWeights[n][i, j];
                    }
                }
            }
        }


        /// <summary>
        /// 入力を入れてみて結果を計算
        /// </summary>
        /// <param name="input"></param>
        public double[] InputData(List<double> input)
        {
            for (var i = 0; i < numNeuron[0]; i++)
            {
                outputs[0][i] = input[i];
            }
            for (var n = 1; n < numLayer; n++)
            {
                for (var i = 0; i < numNeuron[n]; i++)
                {
                    states[n][i] = 0;
                    for (var j = 0; j < numNeuron[n - 1]; j++)
                    {
                        states[n][i] += weights[n - 1][j, i] * outputs[n - 1][j];
                    }
                    outputs[n][i] = activatFunc(states[n][i], T);
                }
            }
            return outputs[numLayer - 1];
        }

        /// <summary>
        /// 学習を実行
        /// </summary>
        /// <param name="numRepeat"></param>
        public void Learn(int numRepeat)
        {
            var tryCount = Math.Min(1000, sample.Count);//全件で誤差を確認するのは時間がかかるので最大1000件にする
            var testData = NeuralFunc.GetNSample(sample, tryCount);//誤差をウォッチしていくデータを指定

            //学習はnumRepeat回繰り返す
            for (var cnt = 0; cnt < numRepeat; cnt++)
            {
                var batches = NeuralFunc.SplitToBatch(sample, batchSize);
                foreach (var batchSample in batches)
                {
                    foreach (var data in batchSample)
                    {
                        var dEdy = new double[numNeuron[numLayer - 1]];

                        InputData(data.Data);

                        //最上位の微分を計算する
                        for (var i = 0; i < numNeuron[numLayer - 1]; i++)
                        {
                            dEdy[i] = outputs[numLayer - 1][i] - data.Answer[i];
                        }

                        for (var n = numLayer - 1; n > 0; n--)
                        {
                            var delta = new double[numNeuron[n]];

                            for (var i = 0; i < numNeuron[n]; i++)
                            {
                                delta[i] = dEdy[i] * differentialFunc(states[n][i], T);
                            }

                            //n-1からn段目への重みの更新
                            for (var i = 0; i < numNeuron[n - 1]; i++)
                            {
                                for (var j = 0; j < numNeuron[n]; j++)
                                {
                                    var d = e * delta[j] * outputs[n - 1][i];
                                    deltaWeights[n - 1][i, j] -= d;
                                }
                            }

                            if (n == 1)
                                break; //n=1の時はこれ以上計算する必要がないので終了

                            var dEdyNext = new double[numNeuron[n - 1]];


                            for (var i = 0; i < numNeuron[n - 1]; i++)
                            {
                                dEdyNext[i] = 0;
                                for (var j = 0; j < numNeuron[n]; j++)
                                {
                                    dEdyNext[i] += dEdy[j] * weights[n - 1][i, j] * differentialFunc(states[n][j], T);
                                }
                            }

                            dEdy = dEdyNext;
                        }
                    }

                    AddDeltaWeight();
                    ResetDeltaWeight();
                }

                if ((10 * cnt) % numRepeat == 0)
                {//出力、教師信号、誤差を表示
                    double error = 0;
                    foreach (var data in testData)
                    {
                        InputData(data.Data);
                        for (var i3 = 0; i3 < numNeuron[numLayer - 1]; i3++)
                        {
                            error += Math.Pow(data.Answer[i3] - outputs[numLayer - 1][i3], 2);
                        }
                    }

                    Console.WriteLine("誤差は" + error / sample.Count);
                }
            }
        }

        /// <summary>
        /// サンプルを一括登録
        /// </summary>
        /// <param name="sampleData"></param>
        public void SetSample(IReadOnlyList<TrainingData> sampleData)
        {
            sample = sampleData.ToList();
        }
    }
}

