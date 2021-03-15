using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using NeuralLib;


namespace FastConvNet
{
    public class FullyConnectedLayer : Layer
    {
        private NeuralImage states; //states[n][k] この層のk番目のニューロンの状態
        public double[][,,] weights; //weights[n][k,l] 一つ下の層のk番目のニューロンとこの層のl番目のニューロンの重み
        public NeuralImage inputs;
        private ActiveFunction activeFunc;
        public int numNeuron;
        public double learnRate;
        public FUNC_TYPE funcType;
        private DifferentialActiveFunction differentialFunc;
        private double T = 2;

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
            this.funcType = funcType;

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

        public override Layer Copy()
        {
            var copy = new FullyConnectedLayer(this.numNeuron, this.underWidth, this.underHeight, this.underPlane, learnRate, funcType);

            for (var p = 0; p < underPlane; p++)
            {
                for (var i = 0; i < underHeight; i++)
                {
                    for (var j = 0; j < underWidth; j++)
                    {

                        for (var k = 0; k < numNeuron; k++)
                        {
                            copy.weights[p][i, j, k] = this.weights[p][i, j, k];

                        }
                    }
                }
            }
            return copy;
        }

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
    }

    public class MaxPoolingLayer : Layer
    {
        public NeuralImage inputs;
        public int filterSize;
        public int stride;
        public Dictionary<(int i, int ypos, int xpos), (int y, int x)> argMaxMap;
        public double learnRate;

        public MaxPoolingLayer(int filterSize, int stride, Layer lastLayer) 
            : this(filterSize, stride, lastLayer.nextWidth, lastLayer.nextHeight, lastLayer.nextPlane) { }

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

        public override NeuralImage CalcOutput(NeuralImage input)
        {
            inputs = input;
            (outputs, argMaxMap) = input.ApplyMax(filterSize, stride);
            return outputs;
        }

        public override Layer Copy()
        {
            var copy = new MaxPoolingLayer(filterSize, stride, underWidth, underHeight, underPlane);
            return copy;
        }

        public override void MergeDeltaWeights(List<Layer> layers)
        {
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
        public double learnRate;
        public int stride;

        public ConvolutionalLayer(int numFilter, int filterSize, int stride, Layer lastLayer, double learnRate = 0.001)
            : this(numFilter, filterSize, stride, lastLayer.nextWidth, lastLayer.nextHeight, lastLayer.nextPlane, learnRate) { }

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

                    filters[i][j] = NeuralFunc.BoxMuler(0, (double)Math.Sqrt(1.0 / filterPix));
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
            outputs = input.ApplyFilters(filters, filterSize, this.stride);
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

        public override Layer Copy()
        {
            var copy = new ConvolutionalLayer(numFilter, filterSize, stride, underWidth, underHeight, underPlane, learnRate, isFirstLayer);
            for (var i = 0; i < numFilter; i++)
            {
                for (var s = 0; s < filterPix; s++)
                {
                    copy.filters[i][s] = this.filters[i][s];
                }
            }
            return copy;
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
    }

    public class SoftMaxLayer : Layer
    {
        private double[] states;
        private double[] outputArray;
        public double[][,,] weights; //weights[n][k,l] 一つ下の層のk番目のニューロンとこの層のl番目のニューロンの重み
        public NeuralImage inputs;
        public int numNeuron;
        public double learnRate;

        public SoftMaxLayer(int numNeuron, Layer lastLayer, double learnRate) 
            : this(numNeuron, lastLayer.nextWidth, lastLayer.nextHeight, lastLayer.nextPlane, learnRate) { }

        public SoftMaxLayer(int numNeuron, int underWidth, int underHeight, int underPlane, double learnRate = 0.001)
        {
            this.underWidth = underWidth;
            this.underHeight = underHeight;
            this.underPlane = underPlane;
            this.numNeuron = numNeuron;
            nextWidth = numNeuron;
            nextHeight = 1;
            nextPlane = 1;
            states = new double[numNeuron];
            outputArray = new double[numNeuron];
            this.learnRate = learnRate;

            dEdyNext = new double[underPlane][,];
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
                            weights[i][j, k, l] = (double)NeuralFunc.BoxMuler(0,
                                    Math.Sqrt(2.0 / (double)(underHeight * underWidth * underPlane))) / 100;
                            deltaWeights[i][j, k, l] = 0;
                        }
                    }
                }
            }
        }

        public override NeuralImage CalcOutput(NeuralImage input)
        {
            inputs = input;
            for (var l = 0; l < numNeuron; l++)
            {
                states[l] = 0;
                for (var i = 0; i < underPlane; i++)
                {
                    for (var j = 0; j < underHeight; j++)
                    {
                        for (var k = 0; k < underWidth; k++)
                        {
                            states[l] += weights[i][j, k, l] * input.data[i][j, k];
                        }
                    }
                }
            }
            outputArray = NeuralFunc.SoftMax(states);
            for (var i = 0; i < numNeuron; i++)
            {
                outputs.data[0][0, i] = outputArray[i];
            }
            return outputs;
        }

        public override void CalcBackPropagation(NeuralImage dEdy)
        {
            var dydx = NeuralFunc.DSoftMax(outputArray);

            for (var p = 0; p < underPlane; p++)
            {
                for (var i = 0; i < underHeight; i++)
                {
                    for (var j = 0; j < underWidth; j++)
                    {
                        for (var k = 0; k < numNeuron; k++)
                        {
                            var delta = 0.0;
                            for (var m = 0; m < numNeuron; m++)
                            {
                                delta = dEdy.data[0][0, m] * dydx[k, m];
                            }
                            var d = delta * inputs.data[p][i, j];
                            deltaWeights[p][i, j, k] += d;
                        }
                    }
                }
            }

            dEdyNext = new double[underPlane][,];

            for (var p = 0; p < underPlane; p++)
            {
                dEdyNext[p] = new double[underHeight, underWidth];
                for (var i = 0; i < underHeight; i++)
                {
                    for (var j = 0; j < underWidth; j++)
                    {
                        dEdyNext[p][i, j] = 0;
                        for (var k = 0; k < numNeuron; k++)
                        {
                            var sigma = 0.0;
                            for (var m = 0; m < numNeuron; m++)
                            {
                                sigma += weights[p][i, j, m] * dydx[m, k];
                            }

                            dEdyNext[p][i, j] += dEdy.data[0][0, k] * sigma;
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
        public override void MergeDeltaWeights(List<Layer> layers)
        {
            foreach (var softLayer in layers.Cast<SoftMaxLayer>())
            {
                for (var p = 0; p < underPlane; p++)
                {
                    for (var i = 0; i < underHeight; i++)
                    {
                        for (var j = 0; j < underWidth; j++)
                        {
                            for (var k = 0; k < numNeuron; k++)
                            {
                                deltaWeights[p][i, j, k] += softLayer.deltaWeights[p][i, j, k];
                            }
                        }
                    }
                }
            }
        }

        public override Layer Copy()
        {
            var copy = new SoftMaxLayer(this.numNeuron, this.underWidth, this.underHeight, this.underPlane, learnRate);

            for (var p = 0; p < underPlane; p++)
            {
                for (var i = 0; i < underHeight; i++)
                {
                    for (var j = 0; j < underWidth; j++)
                    {
                        for (var k = 0; k < numNeuron; k++)
                        {
                            copy.weights[p][i, j, k] = this.weights[p][i, j, k];
                        }
                    }
                }
            }
            return copy;
        }
    }

    public class ConvolutionalMnLayer : Layer
    {
        public double[,][] filters;
        public double[,][] deltaWeights;
        public NeuralImage inputs;
        public int filterSize;
        public int filterPix;
        public int numFilter;
        public (int, int)[][] connectionMapX;
        public (int, int)[][] connectionMapY;
        public bool isFirstLayer = false;
        public int paddingLength = 0;
        public double learnRate;
        public int stride;


        public ConvolutionalMnLayer(int numNextPlane, int filterSize, int stride, int underWidth, int underHeight, int underPlane, bool isFirstLayer = false, int paddingLength = 0, double learnRate = 0.001)
        {
            this.numFilter = numNextPlane * underPlane;
            this.filterSize = filterSize;
            this.learnRate = learnRate;
            this.paddingLength = paddingLength;

            this.underWidth = underWidth + 2 * paddingLength;
            this.underHeight = underHeight + 2 * paddingLength;
            this.underPlane = underPlane;
            this.stride = stride;
            this.dEdyNext = new double[underPlane][,];
            this.isFirstLayer = isFirstLayer;

            connectionMapX = NeuralFunc.CreateConnectionMap(this.underWidth, filterSize, stride);
            connectionMapY = NeuralFunc.CreateConnectionMap(this.underHeight, filterSize, stride);

            nextWidth = (int)Math.Floor((this.underWidth - filterSize + stride) / (double)stride);
            nextHeight = (int)Math.Floor((this.underHeight - filterSize + stride) / (double)stride);
            nextPlane = numNextPlane;

            outputs = new NeuralImage(nextPlane, nextWidth, nextHeight);
            filters = new double[underPlane, nextPlane][];
            deltaWeights = new double[underPlane, nextPlane][];


            filterPix = filterSize * filterSize;

            for (int k = 0; k < underPlane; k++)
            {
                for (var i = 0; i < nextPlane; i++)
                {
                    filters[k, i] = new double[filterPix];
                    deltaWeights[k, i] = new double[filterPix];

                    for (var j = 0; j < filterPix; j++)
                    {
                        filters[k, i][j] = NeuralFunc.BoxMuler(0, 1.0 / filterPix);
                    }
                }
            }
        }

        public string PrintWeight()
        {
            var resultStr = "";
            for (int i = 0; i < nextPlane; i++)
            {
                for (int p = 0; p < underPlane; p++)
                {
                    resultStr += $"filter {i}:{p}\n";
                    for (int j = 0; j < filterSize; j++)
                    {
                        for (int k = 0; k < filterSize; k++)
                        {
                            resultStr += filters[p, i][j * filterSize + k].ToString("F3") + ",";
                        }

                        resultStr += ";\n";
                    }
                }
            }
            return resultStr;
        }

        public override NeuralImage CalcOutput(NeuralImage input)
        {
            if (paddingLength != 0)
            {
                inputs = input.AddPaddingImage(paddingLength);
            }
            else
            {
                inputs = input;
            }
            outputs = inputs.MatrixAppry(filters, filterSize, stride, nextPlane);
            return outputs;
        }


        public override void CalcBackPropagation(NeuralImage dEdy)
        {
            for (var p = 0; p < nextPlane; p++)
            {
                for (var q = 0; q < underPlane; q++)
                {
                    for (var i = 0; i < dEdy.height; i++)
                    {
                        for (var j = 0; j < dEdy.width; j++)
                        {
                            var alpha = dEdy.data[p][i, j];
                            if (alpha == 0.0)
                            {
                                continue;
                            }

                            for (var s = 0; s < filterSize; s++)
                            {
                                var y = stride * i + s;
                                var x = stride * j;
                                var ptr = s * filterSize;
                                for (var t = 0; t < filterSize; t++)
                                {
                                    deltaWeights[q, p][ptr++] += alpha * inputs.data[q][y, x++];
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
                var dEdyNextP = new double[underHeight - 2 * paddingLength, underWidth - 2 * paddingLength];
                for (var q = 0; q < nextPlane; q++)
                {

                    for (var i = paddingLength; i < underHeight - paddingLength; i++)
                    {
                        var connectionY = connectionMapY[i];
                        for (var j = paddingLength; j < underWidth - paddingLength; j++)
                        {
                            var sum = 0.0;
                            var connectionX = connectionMapX[j];
                            foreach (var (y, t) in connectionY)
                            {
                                var ptr = t * filterSize;

                                foreach (var (x, s) in connectionX)
                                {
                                    sum += dEdy.data[q][y, x] * filters[p, q][ptr + s];
                                }
                            }

                            dEdyNextP[i - paddingLength, j - paddingLength] += sum;
                        }
                    }
                }

                dEdyNext[p] = dEdyNextP;
            }
        }

        public override void updateWeight()
        {
            var sum = 0.0;
            for (var i = 0; i < underPlane; i++)
            {
                for (var j = 0; j < nextPlane; j++)
                {
                    for (var s = 0; s < filterPix; s++)
                    {
                        filters[i, j][s] -= learnRate * deltaWeights[i, j][s];
                        deltaWeights[i, j][s] = 0;
                        sum += filters[i, j][s] * filters[i, j][s];
                    }
                }
            }
        }

        public override Layer Copy()
        {
            var originalWidth = underWidth - paddingLength * 2;
            var originalHeight = underHeight - paddingLength * 2;
            var copy = new ConvolutionalMnLayer(nextPlane, filterSize, stride, originalWidth, originalHeight, underPlane, isFirstLayer, paddingLength, learnRate);
            for (var i = 0; i < underPlane; i++)
            {
                for (var j = 0; j < nextPlane; j++)
                {
                    for (var s = 0; s < filterPix; s++)
                    {
                        copy.filters[i, j][s] = filters[i, j][s];
                    }
                }
            }
            return copy;
        }

        public override void MergeDeltaWeights(List<Layer> layers)
        {
            foreach (var convLayer in layers.Cast<ConvolutionalMnLayer>())
            {
                for (var i = 0; i < underPlane; i++)
                {
                    for (var j = 0; j < nextPlane; j++)
                    {
                        for (var s = 0; s < filterPix; s++)
                        {
                            deltaWeights[i, j][s] += convLayer.deltaWeights[i, j][s];
                        }
                    }
                }
            }
        }
    }

    public class ConvolutionalMnLayer3x3 : Layer
    {
        public Vector3[,][] filters;
        public Vector3[,][] deltaWeights;
        public NeuralImageSingle inputs;
        public int filterSize = 3;
        public int filterPix;
        public int numFilter;
        public (int, int)[][] connectionMapY;
        public bool isFirstLayer = false;
        public int paddingLength = 1;
        public int stride = 1;

        public float learnRate;

        public ConvolutionalMnLayer3x3(int numNextPlane, Layer lastLayer, double learnRate = 0.001)
        : this(numNextPlane, lastLayer.nextWidth, lastLayer.nextHeight, lastLayer.nextPlane, false, learnRate) { }
        public ConvolutionalMnLayer3x3(int numNextPlane, int underWidth, int underHeight, int underPlane, bool isFirstLayer = false, double learnRate = 0.001)
        {
            this.numFilter = numNextPlane * underPlane;
            this.underWidth = underWidth + 2 * paddingLength;
            this.underHeight = underHeight + 2 * paddingLength;
            this.underPlane = underPlane;
            this.learnRate = (float)learnRate;
            this.dEdyNext = new double[underPlane][,];
            this.isFirstLayer = isFirstLayer;
            connectionMapY = NeuralFunc.CreateConnectionMap(this.underHeight, filterSize, stride);

            nextWidth = (int)Math.Floor((this.underWidth - filterSize + stride) / (double)stride);
            nextHeight = (int)Math.Floor((this.underHeight - filterSize + stride) / (double)stride);
            nextPlane = numNextPlane;

            outputs = new NeuralImage(nextPlane, nextWidth, nextHeight);
            filters = new Vector3[underPlane, nextPlane][];
            deltaWeights = new Vector3[underPlane, nextPlane][];

            filterPix = filterSize * filterSize;

            for (int k = 0; k < underPlane; k++)
            {
                for (var i = 0; i < nextPlane; i++)
                {
                    filters[k, i] = new Vector3[3];
                    deltaWeights[k, i] = new Vector3[3];

                    for (var j = 0; j < 3; j++)
                    {

                        filters[k, i][j] = new Vector3((float)NeuralFunc.BoxMuler(0, 1.0 / filterPix),
                            (float)NeuralFunc.BoxMuler(0, 1.0 / filterPix),
                            (float)NeuralFunc.BoxMuler(0, 1.0 / filterPix));
                    }
                }
            }
        }


        public string PrintWeight()
        {
            var resultStr = "";
            for (int i = 0; i < nextPlane; i++)
            {
                for (int p = 0; p < underPlane; p++)
                {
                    resultStr += $"filter {i}:{p}\n";
                    for (int j = 0; j < filterSize; j++)
                    {
                        for (int k = 0; k < filterSize; k++)
                        {
                            resultStr += filters[p, i][j * filterSize + k].ToString("F3") + ",";
                        }
                        resultStr += ";\n";
                    }
                }
            }
            return resultStr;
        }

        public override NeuralImage CalcOutput(NeuralImage input)
        {
            inputs = input.AddPaddingImageS(paddingLength);
            outputs = new NeuralImage(inputs.MatrixAppry3x3(filters, nextPlane));
            return outputs;
        }


        public override unsafe void CalcBackPropagation(NeuralImage dEdy)
        {
            var vec = new Vector3();
            var v1 = new Vector3();
            var v2 = new Vector3();
            var v3 = new Vector3();
            for (var p = 0; p < nextPlane; p++)
            {
                for (var q = 0; q < underPlane; q++)
                {
                    var deltaWeightsQP = deltaWeights[q, p];
                    fixed (float* inputsDataQ = &inputs.data[q][0, 0])
                    {
                        for (var i = 0; i < dEdy.height; i++)
                        {
                            for (var j = 0; j < dEdy.width; j++)
                            {
                                var alpha = (float)dEdy.data[p][i, j];
                                if (alpha == 0.0)
                                {
                                    continue;
                                }
                                var ptr = i * inputs.width + j;
                                v1.X = inputsDataQ[ptr++]; v1.Y = inputsDataQ[ptr++]; v1.Z = inputsDataQ[ptr];
                                ptr = (i + 1) * inputs.width + j;
                                v2.X = inputsDataQ[ptr++]; v2.Y = inputsDataQ[ptr++]; v2.Z = inputsDataQ[ptr];
                                ptr = (i + 2) * inputs.width + j;
                                v3.X = inputsDataQ[ptr++]; v3.Y = inputsDataQ[ptr++]; v3.Z = inputsDataQ[ptr];
                                deltaWeightsQP[0] += alpha * v1;
                                deltaWeightsQP[1] += alpha * v2;
                                deltaWeightsQP[2] += alpha * v3;
                            }
                        }
                    }
                }
            }

            if (isFirstLayer)
            {
                return;
            }

            var dEdyS = new NeuralImageSingle(dEdy);


            for (var p = 0; p < underPlane; p++)
            {
                var dEdyNextP = new double[underHeight - 2 * paddingLength, underWidth - 2 * paddingLength];
                for (var q = 0; q < nextPlane; q++)
                {
                    var filter = filters[p, q];
                    fixed (float* dEdyData = &dEdyS.data[q][0, 0])
                    {
                        for (var i = paddingLength; i < underHeight - paddingLength; i++)
                        {
                            var connectionY = connectionMapY[i];
                            for (var j = paddingLength; j < underWidth - paddingLength; j++)
                            {
                                var sum = 0.0f;
                                foreach (var (y, t) in connectionY)
                                {
                                    var end = Math.Min(j, nextWidth - 1);
                                    var start = Math.Max(0, j - filterSize + 1);
                                    var n = end - start;
                                    var ptr = y * dEdyS.width + start;

                                    if (n == 1)
                                    {
                                        var ptrStart = Math.Min(j, filterSize - 1);
                                        switch (ptrStart)
                                        {
                                            case 1:
                                                vec.Z = 0;
                                                vec.Y = dEdyData[ptr++];
                                                vec.X = dEdyData[ptr];
                                                break;
                                            case 2:
                                                vec.Z = dEdyData[ptr++];
                                                vec.Y = dEdyData[ptr];
                                                vec.X = 0;
                                                break;
                                        }
                                    }
                                    else if (n == 2)
                                    {
                                        vec.Z = dEdyData[ptr++];
                                        vec.Y = dEdyData[ptr++];
                                        vec.X = dEdyData[ptr];
                                    }
                                    sum += Vector3.Dot(filter[t], vec);
                                }
                                dEdyNextP[i - paddingLength, j - paddingLength] += sum;
                            }
                        }
                    }
                }
                dEdyNext[p] = dEdyNextP;
            }
        }

        public override void updateWeight()
        {
            for (var i = 0; i < underPlane; i++)
            {
                for (var j = 0; j < nextPlane; j++)
                {
                    for (var s = 0; s < 3; s++)
                    {
                        filters[i, j][s] -= learnRate * deltaWeights[i, j][s];
                        deltaWeights[i, j][s] = new Vector3(0, 0, 0);
                    }
                }
            }
        }

        public override Layer Copy()
        {
            var copy = new ConvolutionalMnLayer3x3(nextPlane, underWidth - paddingLength * 2, underHeight - paddingLength * 2, underPlane, isFirstLayer, learnRate);
            for (var i = 0; i < underPlane; i++)
            {
                for (var j = 0; j < nextPlane; j++)
                {
                    for (var s = 0; s < 3; s++)
                    {
                        copy.filters[i, j][s] = this.filters[i, j][s];
                    }
                }
            }
            return copy;
        }

        public override void MergeDeltaWeights(List<Layer> layers)
        {
            foreach (var convLayer in layers.Cast<ConvolutionalMnLayer3x3>())
            {
                for (var i = 0; i < underPlane; i++)
                {
                    for (var j = 0; j < nextPlane; j++)
                    {
                        for (var s = 0; s < 3; s++)
                        {
                            deltaWeights[i, j][s] += convLayer.deltaWeights[i, j][s];
                        }
                    }
                }
            }
        }
    }

    public class ActivationLayer : Layer
    {
        public NeuralImage inputs;
        public double learnRate;
        private ActiveFunction activeFunc;
        private DifferentialActiveFunction differentialFunc;
        private FUNC_TYPE funcType;

        public ActivationLayer(Layer lastLayer, double learnRate = 0.001, FUNC_TYPE funcType = FUNC_TYPE.ReLU) 
            : this(lastLayer.nextWidth, lastLayer.nextHeight, lastLayer.nextPlane, learnRate, funcType){ }
        public ActivationLayer(int underWidth, int underHeight, int underPlane, double learnRate = 0.001, FUNC_TYPE funcType = FUNC_TYPE.ReLU)
        {
            this.dEdyNext = new double[underPlane][,];
            this.underPlane = underPlane;
            this.underWidth = underWidth;
            this.underHeight = underHeight;
            this.funcType = funcType;
            this.nextHeight = underHeight;
            this.nextWidth = underWidth;
            this.nextPlane = underPlane;
            this.learnRate = learnRate;

            outputs = new NeuralImage(nextPlane, nextWidth, nextHeight);
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
            return input.ApplyFunction(activeFunc, 0);
        }

        public override void CalcBackPropagation(NeuralImage dEdy)
        {
            for (int p = 0; p < underPlane; p++)
            {
                dEdyNext[p] = new double[underHeight, underWidth];
                for (int y = 0; y < underHeight; y++)
                {
                    for (int x = 0; x < underWidth; x++)
                    {
                        dEdyNext[p][y, x] = dEdy.data[p][y, x] * differentialFunc(inputs.data[p][y, x], 0);
                    }
                }
            }
        }

        public override void updateWeight()
        {
        }

        public override Layer Copy()
        {
            return new ActivationLayer(underWidth, underHeight, underPlane, learnRate, funcType);
        }

        public override void MergeDeltaWeights(List<Layer> layers)
        {
        }
    }
}
