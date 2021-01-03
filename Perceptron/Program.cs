﻿using System;
using System.Linq;
using NeuralLib;

namespace Perceptron
{
    class Program
    {
        static void Main(string[] args)
        {
            var data = MnistLoader.ReadData(2000, "c:/data/mnist/train-images-idx3-ubyte.gz", "c:/data/mnist/train-labels-idx1-ubyte.gz");
            //ThreeLayerPerceptron perceptron = new ThreeLayerPerceptron(28*28,100, 10);
            var perceptron = new ManyLayerPerceptron2(new[]{28*28,50,10});
            perceptron.SetSample(data.ToList());
            perceptron.Learn(500);

            var test = MnistLoader.ReadData(1000, "c:/data/mnist/t10k-images-idx3-ubyte.gz", "c:/data/mnist/t10k-labels-idx1-ubyte.gz");
            var successCount = 0.0;
            var totalCount = 0.0;
            foreach(var testData in test)
            {
                var result = perceptron.InputData(testData.Data.ToArray());
                var argMax = result.Select((d, i) => (d, i)).OrderByDescending(d => d.d).First().i;
                var answer = testData.Answer.Select((d, i) => (d, i)).OrderByDescending(d => d.d).First().i;
                if(argMax == answer)
                {
                    successCount++;
                }
                totalCount++;
            }
            Console.WriteLine($"recognition rate={successCount / totalCount}");
            Console.ReadLine();
        }
    }
}
