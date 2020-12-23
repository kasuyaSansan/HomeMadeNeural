using System;
using System.Collections.Generic;
using System.Linq;
using NeuralLib;

namespace Perceptron
{
    public class FullyConnectedLayer
    {
        private double[] states; 
        public double[] outputs; 
        internal double[,] weights; 
        private double[,] deltaWeights;
        private double[] inputs;
        private int numNeuron;
        private int numNeuronUnderLayer; 
        internal double[] dEdyNext;
        private delegate double activationFunction(double x, double param);
        private activationFunction activatFunc;
        private delegate double differentialActivationFunction(double x, double param);
        private differentialActivationFunction differentialFunc;
        public double e = 0.001;
        private double T = 2;

        public FullyConnectedLayer(int numNeuron, int numNeuronUnderLayer, double learnRate = 0.001, FUNC_TYPE funcType = FUNC_TYPE.Sigmoid)
        {
            this.numNeuron = numNeuron;
            this.numNeuronUnderLayer = numNeuronUnderLayer;
            e = learnRate;
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

            states = new double[numNeuron];
            outputs = new double[numNeuron];
            weights = new double[numNeuronUnderLayer, numNeuron];
            deltaWeights = new double[numNeuronUnderLayer, numNeuron];

            //重みを乱数で初期化
            var r = new Random();

            for (var i = 0; i < numNeuronUnderLayer; i++)
            {
                for (var j = 0; j < numNeuron; j++)
                {
                    weights[i, j] =
                        NeuralFunc.BoxMuler(0, Math.Sqrt(2.0 / numNeuron)); 
                }
            }
        }

        public void CalcOutput(double[] input)
        {
            inputs = input.ToArray();

            for (var i = 0; i < numNeuron; i++)
            {
                states[i] = 0;
                for (var j = 0; j < input.Length; j++)
                {
                    states[i] += weights[j, i] * input[j];
                }

                outputs[i] = activatFunc(states[i], T);
            }
        }

        public void CalcBackPropagation(double[] dEdy)
        {
            var delta = new double[numNeuron];

            for (var i = 0; i < numNeuron; i++)
            {
                delta[i] = dEdy[i] * differentialFunc(states[i], T);
            }

            //n-1からn段目への重みの更新
            for (var i = 0; i < numNeuronUnderLayer; i++)
            {
                for (var j = 0; j < numNeuron; j++)
                {

                    var d = e * delta[j] * inputs[i];

                    deltaWeights[i, j] -= d;
                }
            }


            dEdyNext = new double[numNeuronUnderLayer];

            for (var i = 0; i < numNeuronUnderLayer; i++)
            {
                dEdyNext[i] = 0;
                for (var j = 0; j < numNeuron; j++)
                {
                    dEdyNext[i] += dEdy[j] * weights[i, j] * differentialFunc(states[j], T);
                }
            }
        }

        public void UpdateWeight()
        {
            var absMax = 0.0;

            for (var i = 0; i < numNeuronUnderLayer; i++)
            {
                for (var j = 0; j < numNeuron; j++)
                {
                    absMax = Math.Max(Math.Abs(deltaWeights[i, j]), absMax);
                }
            }

            var coef = 1.0;

            var ratio = absMax / Math.Sqrt(2.0 / numNeuron);

            if (ratio > 0.05)
            {
                coef = 0.05 / ratio;
            }

            for (var i = 0; i < numNeuronUnderLayer; i++)
            {
                for (var j = 0; j < numNeuron; j++)
                {
                    weights[i, j] += coef * deltaWeights[i, j];
                    deltaWeights[i, j] = 0;
                }
            }
        }
    }

    public class ManyLayerPerceptron2
    {
        private int numOutputLayer;
        public List<FullyConnectedLayer> Layers;
        private List<TrainingData> sample = new List<TrainingData>();//学習サンプル
        private int numLayer;

        private int batchSize = 100;

        /// <summary>
        /// コンストラクタ
        /// </summary>
        /// <param name="numNeurons"></param>
        /// <param name="funcType"></param>
        public ManyLayerPerceptron2(IReadOnlyList<int> numNeurons, FUNC_TYPE funcType = FUNC_TYPE.Sigmoid)
        {
            Layers = new List<FullyConnectedLayer>();
            numLayer = numNeurons.Count;
            numOutputLayer = numNeurons.Last();
            var numNeuronsAnd0 = numNeurons.ToList();
            numNeuronsAnd0.Add(0);
            for (var i = 0; i < numNeurons.Count - 1; i++)
            {
                Layers.Add(new FullyConnectedLayer(numNeurons[i + 1], numNeurons[i], 0.001, funcType));
            }
        }


        /// <summary>
        /// 入力を入れてみて結果を計算
        /// </summary>
        /// <param name="input"></param>
        public double[] InputData(double[] input)
        {
            Layers[0].CalcOutput(input);
            for (var i = 1; i < numLayer - 1; i++)
            {
                Layers[i].CalcOutput(Layers[i - 1].outputs);
            }
            return Layers.Last().outputs;
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
                        var dEdy = new double[numOutputLayer];
                        InputData(data.Data.ToArray());
                        var dEdyLast = new double[numOutputLayer];

                        //最上位の微分を計算する
                        for (var i = 0; i < numOutputLayer; i++)
                        {
                            dEdyLast[i] = Layers.Last().outputs[i] - data.Answer[i];
                        }

                        Layers.Last().CalcBackPropagation(dEdyLast);
                        for (var n = Layers.Count - 2; n >= 0; n--)
                        {
                            Layers[n].CalcBackPropagation(Layers[n + 1].dEdyNext);
                        }
                    }

                    foreach (var layer in Layers)
                    {
                        layer.UpdateWeight();
                    }
                }

                if ((10 * cnt) % numRepeat == 0)
                {//出力、教師信号、誤差を表示
                    double error = 0;
                    foreach (var data in testData)
                    {
                        var outputs = InputData(data.Data.ToArray());
                        for (var i3 = 0; i3 < numOutputLayer; i3++)
                        {
                            error += Math.Pow(data.Answer[i3] - outputs[i3], 2);
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
