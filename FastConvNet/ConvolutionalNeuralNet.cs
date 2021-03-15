using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using NeuralLib;

namespace FastConvNet
{
    public abstract class Layer
    {
        public abstract NeuralImage CalcOutput(NeuralImage input);
        public abstract void CalcBackPropagation(NeuralImage dEdy);
        public abstract void updateWeight();
        public abstract void MergeDeltaWeights(List<Layer> layers);
        public abstract Layer Copy();
        public int underWidth;
        public int underHeight;
        public int underPlane;
        public int nextWidth;
        public int nextHeight;
        public int nextPlane;
        public double[][,,] deltaWeights;
        public double[][,] dEdyNext;
        public NeuralImage outputs;
    }

    /// <summary>
    /// このクラスのlayersにLayerクラスのインスタンスを追加していってニューラルネットを作る
    /// </summary>
    public class ConvolutionalNeuralNet
    {
        public List<Layer> layers = new List<Layer>();
        public List<TrainingImageData> sample = new List<TrainingImageData>();//学習サンプル
        public int numOutputLayer;
        public double lastError = 1.0;

        public ConvolutionalNeuralNet()
        {
        }

        public NeuralImage InputImage(NeuralImage img)
        {
            var input = img;
            for (int i = 0; i < layers.Count; i++)
            {
                input = layers[i].CalcOutput(input);
            }
            return input;
        }

        public void BackProp(NeuralImage img, int answer)
        {
            var input = img;
            for (int i = 0; i < layers.Count; i++)
            {
                input = layers[i].CalcOutput(input);
            }

            var error = input.data[0][0, 0] - answer;
            var back = new NeuralImage(new double[] { error }, 1, 1);
            for (int i = layers.Count - 1; i >= 0; i--)
            {
                layers[i].CalcBackPropagation(back);
                back = new NeuralImage(layers[i].dEdyNext);
            }
        }

        /// <summary>
        /// 学習を実行
        /// </summary>
        /// <param name="numRepeat"></param>
        public void Learn(int numRepeat, int batchSize = 128)
        {
            numOutputLayer = layers.Last().nextHeight * layers.Last().nextWidth * layers.Last().nextPlane;
            var watch = Stopwatch.StartNew();
            var tryCount = Math.Min(1000, sample.Count);
            var testData = NeuralFunc.GetNSample(sample, tryCount);

            //学習はnumRepeat回繰り返す
            for (var cnt = 0; cnt < numRepeat; cnt++)
            {
                var batchSamples = NeuralFunc.SplitToBatch(sample, batchSize);
                foreach (var batchSample in batchSamples)
                {
                    foreach (var trainingData in batchSample)
                    {
                        var dEdy = new double[numOutputLayer];

                        InputImage(trainingData.Data);
                        var dEdyLast = new double[numOutputLayer];

                        //最上位の微分を計算する
                        for (var i = 0; i < numOutputLayer; i++)
                        {
                            dEdyLast[i] = layers.Last().outputs.data[0][0, i] - trainingData.Answer.data[0][0, i];
                        }

                        var back = new NeuralImage(dEdyLast, numOutputLayer, 1);
                        for (int i = layers.Count - 1; i >= 0; i--)
                        {
                            layers[i].CalcBackPropagation(back);
                            if (i > 0)
                            {
                                back = new NeuralImage(layers[i].dEdyNext);
                            }
                        }
                    }

                    foreach (var layer in layers)
                    {
                        layer.updateWeight();
                    }
                }
                if ((5 * cnt) % numRepeat == 0 || cnt % 10000 == 0)
                {//出力、教師信号、誤差を表示

                    double error = 0;
                    Console.WriteLine("avg time:" + watch.ElapsedMilliseconds / (cnt + 1));
                    for (var tp = 0; tp < tryCount; tp++)
                    {
                        var outputs = InputImage(testData[tp].Data);
                        for (var i = 0; i < numOutputLayer; i++)
                        {
                            error += (double)Math.Pow(testData[tp].Answer.data[0][0, i] - outputs.data[0][0, i], 2);
                        }
                    }

                    lastError = error / tryCount;
                    Console.WriteLine("誤差は" + lastError);
                }
            }
        }


        public ConvolutionalNeuralNet Copy()
        {
            var convNet = new ConvolutionalNeuralNet();
            foreach (var layer in layers)
            {
                convNet.layers.Add(layer.Copy());
            }
            convNet.numOutputLayer = numOutputLayer;
            return convNet;
        }


        /// <summary>
        /// 並列に学習を実行
        /// </summary>
        /// <param name="numRepeat"></param>
        public void LearnParallel(int numRepeat, int numParallel, int batchSize = 128)
        {
            numOutputLayer = layers.Last().nextHeight * layers.Last().nextWidth * layers.Last().nextPlane;
            var watch = Stopwatch.StartNew();
            var tryCount = Math.Min(1000, sample.Count);
            var testData = NeuralFunc.GetNSample(sample, tryCount);

            //学習はnumRepeat回繰り返す
            for (var cnt = 0; cnt < numRepeat; cnt++)
            {
                var batchSamples = NeuralFunc.SplitToBatch(sample, batchSize);
                foreach (var batchSample in batchSamples)
                {
                    var convNets = new ConvolutionalNeuralNet[numParallel];;

                    for (var i = 0; i < convNets.Length; i++)
                    {
                        convNets[i] = Copy();
                    }

                    var numParallelForThisTime = numParallel;
                    if (batchSample.Count % numParallel != 0)
                    {
                        for (var i = 1; i < numParallel; i++)
                        {
                            if (batchSample.Count % i == 0)
                            {
                                numParallelForThisTime = i;
                            }
                        }
                    }

                    var numFor1Thread = batchSample.Count / numParallelForThisTime;
                    Parallel.For(0, numParallelForThisTime, new ParallelOptions() { MaxDegreeOfParallelism = numParallelForThisTime }, threadNo =>
                    {
                        for (var pt = 0; pt < numFor1Thread; pt++)
                        {
                            var dataForThisThread = threadNo * numFor1Thread + pt;
                            convNets[threadNo].InputImage(batchSample[dataForThisThread].Data);
                            var dEdyLast = new double[numOutputLayer];

                            //最上位の微分を計算する
                            for (var i = 0; i < numOutputLayer; i++)
                            {
                                dEdyLast[i] = convNets[threadNo].layers.Last().outputs.data[0][0, i] -
                                              batchSample[dataForThisThread].Answer.data[0][0, i];
                            }

                            var back = new NeuralImage(dEdyLast, numOutputLayer, 1);
                            for (var i = convNets[threadNo].layers.Count - 1; i >= 0; i--)
                            {
                                convNets[threadNo].layers[i].CalcBackPropagation(back);
                                if (i > 0)
                                {
                                    back = new NeuralImage(convNets[threadNo].layers[i].dEdyNext);
                                }
                            }
                        }
                    });
                    
                    for (var i = 0; i < layers.Count; i++)
                    {
                        var sameLayers = convNets.Select(convNet => convNet.layers[i]).ToList();
                        layers[i].MergeDeltaWeights(sameLayers);
                        layers[i].updateWeight();
                    }
                }

                double error = 0;

                Console.WriteLine("avg time:" + watch.ElapsedMilliseconds / (cnt + 1));
                for (var tp = 0; tp < tryCount; tp++)
                {
                    var outputs = InputImage(testData[tp].Data);
                    for (var i3 = 0; i3 < numOutputLayer; i3++)
                    {
                        error += (double)Math.Pow(testData[tp].Answer.data[0][0, i3] - outputs.data[0][0, i3], 2);
                    }
                }

                lastError = error / tryCount;
                Console.WriteLine("誤差は" + lastError);
            }
        }

        /// <summary>
        /// サンプルを登録
        /// </summary>
        /// <param name="sample1"></param>
        /// <param name="teacher1"></param>
        public void SetSample(IReadOnlyCollection<TrainingImageData> trainingImageDataList)
        {
            sample = trainingImageDataList.ToList();
        }
    }
}
