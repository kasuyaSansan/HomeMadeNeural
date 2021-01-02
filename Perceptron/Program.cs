using System;
using System.Linq;
using NeuralLib;

namespace Perceptron
{
    class Program
    {
        static void Main(string[] args)
        {
            var data = MnistLoader.ReadData();
            //ThreeLayerPerceptron perceptron = new ThreeLayerPerceptron(28*28,100, 10);
            var perceptron = new ManyLayerPerceptron2(new[]{28*28,50,10});
            perceptron.SetSample(data.MnistData.ToList());
            perceptron.Learn(100);
        }
    }
}
