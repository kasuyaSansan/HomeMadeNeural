using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using NeuralLib;

namespace ConvNet
{
    public abstract class Layer
    {
        public abstract NeuralImage CalcOutput(NeuralImage input);
        public abstract void CalcBackPropagation(NeuralImage dEdy);
        public abstract void updateWeight();
        public abstract void MergeDeltaWeights(List<Layer> layers);

        public int underWidth;
        public int underHeight;
        public int underPlane;
        public int nextWidth;
        public int nextHeight;
        public int nextPlane;
        public double[][,,] deltaWeights;
        public abstract double learnRate { get; set; }
        public double[][,] dEdyNext;
        public NeuralImage outputs;
    }


    public class FullyConnectedLayer : Layer
    {
        private NeuralImage states; //states[n][k] この層のk番目のニューロンの状態
        public double[][,,] weights; //weights[n][k,l] 一つ下の層のk番目のニューロンとこの層のl番目のニューロンの重み
        public NeuralImage inputs;
        private ActiveFunction activeFunc;
        public int numNeuron;
        public override double learnRate { get; set; }
        private DifferentialActiveFunction differentialFunc;

        private double T = 2;

        public override void MergeDeltaWeights(List<Layer> layers)
        {
            foreach (var fullLayer in layers.Cast<FullyConnectedLayer>())
            {
                for (var p = 0; p < underPlane; p++)
                {
                    for (var i = 0; i < underHeight; i++)
                    {
                        for (var j = 0; j < underWidth; j++)
                        {
                            for (var k = 0; k < numNeuron; k++)
                            {
                                deltaWeights[p][i, j, k] += fullLayer.deltaWeights[p][i, j, k];
                            }
                        }
                    }
                }
            }
        }

        public FullyConnectedLayer(int numNeuron, Layer lastLayer, double learnRate = 0.001, FUNC_TYPE funcType = FUNC_TYPE.Sigmoid)
            : this(numNeuron, lastLayer.nextWidth, lastLayer.nextHeight, lastLayer.nextPlane, learnRate, funcType) { }

        public FullyConnectedLayer(int numNeuron, int underWidth, int underHeight, int underPlane, double learnRate = 0.001, FUNC_TYPE funcType = FUNC_TYPE.Sigmoid)
        {
            this.underWidth = underWidth;
            this.underHeight = underHeight;
            this.underPlane = underPlane;
            this.numNeuron = numNeuron;
            nextWidth = numNeuron;
            nextHeight = 1;
            nextPlane = 1;
            this.learnRate = learnRate;

            dEdyNext = new double[underPlane][,];
            states = new NeuralImage(1, numNeuron, 1);
            outputs = new NeuralImage(1, numNeuron, 1);
            weights = new double[underPlane][,,];
            deltaWeights = new double[underPlane][,,];
            for (var i = 0; i < underPlane; i++)
            {
                weights[i] = new double[underHeight, underWidth, numNeuron];
                deltaWeights[i] = new double[underHeight, underWidth, numNeuron];
                for (var j = 0; j < underHeight; j++)
                {
                    for (var k = 0; k < underWidth; k++)
                    {
                        for (var l = 0; l < numNeuron; l++)
                        {
                            if (funcType != FUNC_TYPE.Sigmoid)
                            {
                                weights[i][j, k, l] = (double)NeuralFunc.BoxMuler(0,
                                    Math.Sqrt(2.0 / (double)(underHeight * underWidth * underPlane)) / 10);
                            }
                            else
                            {
                                weights[i][j, k, l] = (double)NeuralFunc.BoxMuler(0,
                                    2.0 / (double)(underHeight * underWidth * underPlane));
                            }

                            deltaWeights[i][j, k, l] = 0;
                        }
                    }
                }
            }
            if (funcType == FUNC_TYPE.Sigmoid)
            {
                activeFunc = NeuralFunc.Sigmoid;
                differentialFunc = NeuralFunc.Dsigmoid;
            }
            else if (funcType == FUNC_TYPE.LeakyReLU)
            {
                activeFunc = NeuralFunc.LeakyReLU;
                differentialFunc = NeuralFunc.DLeakyReLU;
            }
            else
            {
                activeFunc = NeuralFunc.ReLU;
                differentialFunc = NeuralFunc.DReLU;
            }
        }

        public override NeuralImage CalcOutput(NeuralImage input)
        {
            inputs = input;
            for (var l = 0; l < numNeuron; l++)
            {
                states.data[0][0, l] = 0;
                for (var i = 0; i < underPlane; i++)
                {
                    for (var j = 0; j < underHeight; j++)
                    {
                        for (var k = 0; k < underWidth; k++)
                        {
                            states.data[0][0, l] += weights[i][j, k, l] * input.data[i][j, k];
                        }
                    }
                }

                outputs.data[0][0, l] = activeFunc(states.data[0][0, l], T);
            }

            return outputs;
        }

        public override void CalcBackPropagation(NeuralImage dEdy)
        {

            var delta = new double[numNeuron];
            for (var i = 0; i < numNeuron; i++)
            {
                delta[i] = dEdy.data[0][0, i] * differentialFunc(states.data[0][0, i], T);
            }

            for (var p = 0; p < underPlane; p++)
            {
                for (var i = 0; i < underHeight; i++)
                {
                    for (var j = 0; j < underWidth; j++)
                    {
                        for (var k = 0; k < numNeuron; k++)
                        {
                            var d = delta[k] * inputs.data[p][i, j];
                            deltaWeights[p][i, j, k] += d;
                        }
                    }
                }
            }

            dEdyNext = new double[underPlane][,];

            for (int p = 0; p < underPlane; p++)
            {
                dEdyNext[p] = new double[underHeight, underWidth];
                for (var i = 0; i < underHeight; i++)
                {
                    for (var j = 0; j < underWidth; j++)
                    {
                        dEdyNext[p][i, j] = 0;
                        for (var k = 0; k < numNeuron; k++)
                        {
                            dEdyNext[p][i, j] += dEdy.data[0][0, k] * weights[p][i, j, k] * differentialFunc(states.data[0][0, k], T);
                        }
                    }
                }
            }
        }

        public override void updateWeight()
        {
            var coef = 1.0;
            var absMax = 0.0;

            for (var p = 0; p < underPlane; p++)
            {
                for (var i = 0; i < underHeight; i++)
                {
                    for (var j = 0; j < underWidth; j++)
                    {
                        for (var k = 0; k < numNeuron; k++)
                        {
                            absMax = Math.Max(Math.Abs(deltaWeights[p][i, j, k]), absMax);
                        }
                    }
                }
            }

            var ratio = absMax / (double)Math.Sqrt(2.0 / numNeuron);

            if (ratio > 0.05)
            {
                coef = 0.05 / ratio;
            }

            coef = Math.Min(learnRate, coef);


            for (var p = 0; p < underPlane; p++)
            {
                for (var i = 0; i < underHeight; i++)
                {
                    for (var j = 0; j < underWidth; j++)
                    {
                        for (var k = 0; k < numNeuron; k++)
                        {
                            weights[p][i, j, k] -= coef * deltaWeights[p][i, j, k];
                            deltaWeights[p][i, j, k] = 0;
                        }
                    }
                }
            }
        }
    }

    public class MaxPoolingLayer : Layer
    {
        public NeuralImage inputs;
        public int filterSize;
        public int stride;
        public Dictionary<(int i, int ypos, int xpos), (int y, int x)> argMaxMap;
        public override double learnRate { get; set; }

        public MaxPoolingLayer(int filterSize, int stride, Layer lastLayer) : this(filterSize, stride, lastLayer.nextWidth, lastLayer.nextHeight, lastLayer.nextPlane) { }

        public MaxPoolingLayer(int filterSize, int stride, int underWidth, int underHeight, int underPlane)
        {
            this.filterSize = filterSize;
            this.underWidth = underWidth;
            this.underHeight = underHeight;
            this.underPlane = underPlane;
            var halfLength = (filterSize - 1) / 2;

            nextWidth = (int)Math.Ceiling((underWidth - filterSize + stride) / (double)stride);
            nextHeight = (int)Math.Ceiling((underHeight - filterSize + stride) / (double)stride);
            nextPlane = underPlane;
            this.stride = stride;
            this.dEdyNext = new double[underPlane][,];
        }

        public override void MergeDeltaWeights(List<Layer> layers)
        {
        }

        public override NeuralImage CalcOutput(NeuralImage input)
        {
            inputs = input;
            (outputs, argMaxMap) = input.ApplyMax(filterSize, stride);
            return outputs;
        }

        public override void CalcBackPropagation(NeuralImage dEdy)
        {
            for (var i = 0; i < underPlane; i++)
            {
                var dEdyNextP = new double[underHeight, underWidth];

                for (var ypos = 0; ypos < nextHeight; ypos++)
                {
                    for (var xpos = 0; xpos < nextWidth; xpos++)
                    {
                        var argMax = argMaxMap[(i, ypos, xpos)];
                        dEdyNextP[argMax.y, argMax.x] = dEdy.data[i][ypos, xpos];
                    }
                }

                dEdyNext[i] = dEdyNextP;
            }
        }
        public override void updateWeight() { }
    }


    public class ConvolutionalLayer : Layer
    {
        public double[][] filters;
        public double[][] deltaWeights;
        public NeuralImage inputs;
        public int filterSize;
        public int filterPix;
        public int numFilter;
        public (int, int)[][] connectionMapX;
        public (int, int)[][] connectionMapY;
        public bool isFirstLayer = false;
        public override double learnRate { get; set; }
        public int stride;

        public ConvolutionalLayer(int numFilter, int filterSize, int stride, Layer lastLayer, double learnRate = 0.001)
            :this(numFilter, filterSize, stride, lastLayer.nextWidth, lastLayer.nextHeight, lastLayer.nextPlane, learnRate){}

        public ConvolutionalLayer(int numFilter, int filterSize, int stride, int underWidth, int underHeight, int underPlane, double learnRate = 0.001, bool isFirstLayer = false)
        {
            this.numFilter = numFilter;
            this.filterSize = filterSize;

            this.underWidth = underWidth;
            this.underHeight = underHeight;
            this.underPlane = underPlane;
            this.stride = stride;
            this.dEdyNext = new double[underPlane][,];
            this.isFirstLayer = isFirstLayer;
            this.learnRate = learnRate;

            connectionMapX = NeuralFunc.CreateConnectionMap(underWidth, filterSize, stride);
            connectionMapY = NeuralFunc.CreateConnectionMap(underHeight, filterSize, stride);

            nextWidth = (int)Math.Floor((underWidth - filterSize + stride) / (double)stride);
            nextHeight = (int)Math.Floor((underHeight - filterSize + stride) / (double)stride);
            nextPlane = underPlane * numFilter;

            outputs = new NeuralImage(numFilter * underPlane, nextWidth, nextHeight);
            filters = new double[numFilter][];
            deltaWeights = new double[numFilter][];

            //重みを乱数で初期化
            filterPix = filterSize * filterSize;

            for (var i = 0; i < numFilter; i++)
            {
                filters[i] = new double[filterPix];
                deltaWeights[i] = new double[filterPix];

                for (var j = 0; j < filterPix; j++)
                {

                    filters[i][j] = (double)NeuralFunc.BoxMuler(0, (double)Math.Sqrt(1.0 / filterPix));
                }
            }
        }

        public override void MergeDeltaWeights(List<Layer> layers)
        {
            foreach (var convLayer in layers.Cast<ConvolutionalLayer>())
            {
                for (var i = 0; i < numFilter; i++)
                {
                    for (var s = 0; s < filterPix; s++)
                    {
                        deltaWeights[i][s] += convLayer.deltaWeights[i][s];
                    }
                }
            }
        }

        public void PrintWeight()
        {
            for (int i = 0; i < numFilter; i++)
            {
                Console.WriteLine("filter" + i);
                for (int j = 0; j < filterSize; j++)
                {
                    for (int k = 0; k < filterSize; k++)
                    {
                        Console.Write(filters[i][j * filterSize + k].ToString("F3") + ",");
                    }
                    Console.WriteLine();
                }

            }
        }

        public override NeuralImage CalcOutput(NeuralImage input)
        {
            inputs = input;
            this.outputs = input.ApplyFilters(filters, filterSize, this.stride);
            return outputs;
        }


        public override void CalcBackPropagation(NeuralImage dEdy)
        {
            for (var p = 0; p < numFilter; p++)
            {
                for (var q = 0; q < underPlane; q++)
                {
                    var planeFrom = underPlane * p + q;

                    for (var i = 0; i < dEdy.height; i++)
                    {
                        for (var j = 0; j < dEdy.width; j++)
                        {
                            if (dEdy.data[planeFrom][i, j] == 0.0)
                            {
                                continue;
                            }
                            for (var s = 0; s < filterSize; s++)
                            {
                                var y = stride * i + s;
                                var x = stride * j;
                                for (var t = 0; t < filterSize; t++)
                                {
                                    deltaWeights[p][s * filterSize + t] += dEdy.data[planeFrom][i, j] * inputs.data[q][y, x++];
                                }
                            }
                        }
                    }
                }
            }

            if (isFirstLayer)
            {
                return;
            }

            for (var p = 0; p < underPlane; p++)
            {
                var dEdyNextP = new double[underHeight, underWidth];
                for (var q = 0; q < numFilter; q++)
                {
                    var planeFrom = underPlane * q + p;
                    for (var i = 0; i < underHeight; i++)
                    {
                        var connectionY = connectionMapY[i];
                        for (var j = 0; j < underWidth; j++)
                        {
                            var connectionX = connectionMapX[j];
                            foreach (var (y, t) in connectionY)
                            {
                                foreach (var (x, s) in connectionX)
                                {
                                    dEdyNextP[i, j] += dEdy.data[planeFrom][y, x] * filters[q][t * filterSize + s];
                                }
                            }
                        }
                    }
                }
                dEdyNext[p] = dEdyNextP;
            }
        }

        public override void updateWeight()
        {

            for (var i = 0; i < numFilter; i++)
            {
                for (var s = 0; s < filterPix; s++)
                {
                    filters[i][s] -= learnRate * deltaWeights[i][s];
                    deltaWeights[i][s] = 0;
                }
            }
        }
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
