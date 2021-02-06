using System;
using System.Linq;
using System.Collections.Generic;
using NeuralLib;
using System.IO;

namespace Perceptron
{
    class Program
    {
        static void Main(string[] args)
        {
            var currentPath = Directory.GetCurrentDirectory();
            var mnistPath = currentPath + "/../../../../../NeuralLib/kuzushiji_MNIST";
            var data = MnistLoader.ReadData(4000, Path.Combine(mnistPath, "train-images-idx3-ubyte.gz"), Path.Combine(mnistPath, "train-labels-idx1-ubyte.gz"));

            //ThreeLayerPerceptron perceptron = new ThreeLayerPerceptron(28*28,100, 10);
            var perceptron = new ManyLayerPerceptron2(new[]{28*28,50,10});
            perceptron.SetSample(data);
            perceptron.Learn(300);
            var test = MnistLoader.ReadData(1000, Path.Combine(mnistPath, "t10k-images-idx3-ubyte.gz"), Path.Combine(mnistPath, "t10k-labels-idx1-ubyte.gz"));
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
