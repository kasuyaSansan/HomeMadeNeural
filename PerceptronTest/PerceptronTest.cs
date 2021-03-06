using System;
using System.Collections.Generic;
using NeuralLib;
using NUnit.Framework;
using Perceptron;

namespace PerceptronTest
{
    public class Tests
    {
        [SetUp]
        public void Setup()
        {
        }

        [TestCase(2000)]
        [TestCase(3000)]
        [TestCase(100)]
        public void PercepTronSameResultTest(int numLearn)
        {
            var inputSize = 2;
            var hiddenSize = 3;
            var outputSize = 1;
            var perceptron1 = new ThreeLayerPerceptron(inputSize,hiddenSize,outputSize);
            var perceptron2 = new ManyLayerPerceptron(new []{inputSize,hiddenSize,outputSize}, learnRate:0.001, funcType:FUNC_TYPE.Sigmoid, batchSize:1);
            var perceptron3 = new ManyLayerPerceptron2(new []{inputSize,hiddenSize,outputSize}, funcType: FUNC_TYPE.Sigmoid, batchSize: 1);

            for (var i = 0; i < inputSize; i++)
            {
                for (var j = 0; j < hiddenSize; j++)
                {
                    var weightVal = 1.0 / (i + j + 1);
                    perceptron1.w1to2[i, j] = weightVal;
                    perceptron2.weights[0][i, j] = weightVal;
                    perceptron3.Layers[0]
                        .weights[i, j] = weightVal;
                }
            }

            for (var i = 0; i < hiddenSize; i++)
            {
                for (var j = 0; j < outputSize; j++)
                {
                    var weightVal = 1.0 / (i + j + 1);
                    perceptron1.w2to3[i, j] = weightVal;
                    perceptron2.weights[1][i, j] = weightVal;
                    perceptron3.Layers[1].weights[i, j] = weightVal;
                }
            }

            var data1 = new TrainingData(new[] {0.0, 0.0}, new[] {0.0});
            var data2 = new TrainingData(new[] {1.0, 1.0}, new[] {1.0});

            var trainingData = new List<TrainingData>() {data1, data2};

            perceptron1.SetSample(trainingData);
            perceptron2.SetSample(trainingData);
            perceptron3.SetSample(trainingData);

            perceptron1.Learn(numLearn);
            perceptron2.Learn(numLearn);
            perceptron3.Learn(numLearn);

            var result1 = perceptron1.InputData(new List<double>() {0.5, 0.5});
            var result2 = perceptron2.InputData(new List<double>() { 0.5, 0.5 });
            var result3 = perceptron3.InputData(new [] { 0.5, 0.5 });

            Assert.AreEqual(result1[0].ToString("G5") , result2[0].ToString("G5"));

            Assert.AreEqual(result2[0].ToString("G5"), result3[0].ToString("G5"));
            Console.WriteLine($@"1:{result1[0]}, 2:{result2[0]}, 3:{result3[0]}");
        }
    }
}