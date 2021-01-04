using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using NeuralLib;

namespace Perceptron
{
    public class ThreeLayerPerceptron
    {
        private double[] state2;//第二層の状態
        private double[] state3;//第三層の状態
        private double[] output1;//第一層の出力
        private double[] output2;//第二層の出力
        public double[] output3;//第三層の出力
        internal double[,] w1to2;//第一層から第二層への結合重み
        internal double[,] w2to3;//第二層から第三層への結合重み
        private List<TrainingData> sample = new List<TrainingData>();//学習サンプル
        private int num1, num2, num3;
        public double e;
        private double T = 1.0;
        private int batchSize = 100;

        /// <summary>
        /// コンストラクタ
        /// </summary>
        /// <param name="num1st"></param>
        /// <param name="num2nd"></param>
        /// <param name="num3rd"></param>
        /// <param name="learnRate"></param>
        public ThreeLayerPerceptron(int num1st, int num2nd, int num3rd, double learnRate = 0.001)
        {
            num1 = num1st;
            num2 = num2nd;
            num3 = num3rd;
            e = learnRate;

            //配列の初期化
            output1 = new double[num1];
            state2 = new double[num2];
            output2 = new double[num2];
            state3 = new double[num3];
            output3 = new double[num3];
            w1to2 = new double[num1, num2];
            w2to3 = new double[num2, num3];

            //重みを乱数で初期化
            var r = new Random();
            for (var i = 0; i < num1; i++)
            {
                for (var j = 0; j < num2; j++)
                {
                    w1to2[i, j] = 2 * r.NextDouble() * Math.Pow(-1, r.Next());
                }
            }

            for (var i = 0; i < num2; i++)
            {
                for (var j = 0; j < num3; j++)
                {
                    w2to3[i, j] = 2 * r.NextDouble() * Math.Pow(-1, r.Next());
                }
            }
        }


        /// <summary>
        /// 入力を入れてみて結果を計算
        /// </summary>
        /// <param name="input"></param>
        public double[] InputData(List<double> input)
        {
            for (var i = 0; i < num1; i++)
            {
                output1[i] = input[i];
            }
            for (var i = 0; i < num2; i++)
            {
                state2[i] = 0;
                for (var j = 0; j < num1; j++)
                {
                    state2[i] += w1to2[j, i] * output1[j];
                }
                output2[i] = NeuralFunc.Sigmoid(state2[i], T);
            }

            for (var i = 0; i < num3; i++)
            {
                state3[i] = 0;
                for (var j = 0; j < num2; j++)
                {
                    state3[i] += w2to3[j, i] * output2[j];
                }
                output3[i] = NeuralFunc.Sigmoid(state3[i], T);
            }
            return output3;
        }

        /// <summary>
        /// 学習を実行
        /// </summary>
        /// <param name="numRepeat"></param>
        public void Learn(int numRepeat)
        {
            var watch = Stopwatch.StartNew();
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
                        var delta = new double[num3];
                        var sigma = new double[num2];
                        InputData(data.Data);

                        //微分を計算する
                        for (var i = 0; i < num3; i++)
                        {
                            delta[i] = (data.Answer[i] - output3[i]) * output3[i] * (1 - output3[i]) * T;
                        }

                        //2段目の重みの更新
                        for (var i = 0; i < num2; i++)
                        {
                            for (var j = 0; j < num3; j++)
                            {
                                w2to3[i, j] += e * delta[j] * output2[i];
                            }
                        }

                        for (var i = 0; i < num2; i++)
                        {
                            sigma[i] = 0;
                            for (var j = 0; j < num3; j++)
                            {
                                sigma[i] += delta[j] * w2to3[i, j] * output2[i] * (1 - output2[i]) * T;
                            }
                        }

                        //重みの更新
                        for (var i = 0; i < num1; i++)
                        {
                            for (var j = 0; j < num2; j++)
                            {
                                w1to2[i, j] += e * sigma[j] * output1[i];
                            }
                        }
                    }
                }

                if ((10 * cnt) % numRepeat == 0)
                {//学習中定期的に誤差を表示
                    double error = 0;
                    Console.WriteLine("avg time:" + watch.ElapsedMilliseconds / (cnt + 1));
                    foreach (var test in testData)
                    {
                        InputData(test.Data);
                        for (var i3 = 0; i3 < num3; i3++)
                        {
                            error += (test.Answer[i3] - output3[i3]) * (test.Answer[i3] - output3[i3]);
                        }
                    }
                    Console.WriteLine("誤差は" + (error / sample.Count));
                }
            }
        }


        /// <summary>
        /// サンプルを一括登録
        /// </summary>
        /// <param name="sampleData"></param>
        /// <param name="answer"></param>
        public void SetSample(IReadOnlyList<TrainingData> sampleData)
        {
            sample = sampleData.ToList();
        }
    }
}
