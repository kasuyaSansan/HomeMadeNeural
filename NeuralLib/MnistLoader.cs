using System;
using System.Collections.Generic;
using System.Linq;
using MNIST.IO;

namespace NeuralLib
{
    public enum MnistType
    {
        NORMAL, FASHION
    }

    public class MnistLoader
    {
        private static string normalMnistFileName = "train-images-idx3-ubyte.gz";
        private static string fashionMnistFileName = "t10k-images-idx3-ubyte.gz";

        private static string normalMnistLabelName = "train-labels-idx1-ubyte.gz";



        public static (List<TrainingData> MnistData, List<TrainingData> MnistTest) ReadData(MnistType mnistType = MnistType.NORMAL, int numSample = 2000)
        {
            var useFileName = normalMnistFileName;
            var useLabelFileName = normalMnistLabelName;
            if (mnistType == MnistType.FASHION)
                useFileName = fashionMnistFileName;
            
            var data = FileReaderMNIST.LoadImages($"c:/data/mnist/{useFileName}");
            var label = FileReaderMNIST.LoadLabel($"c:/data/mnist/{useLabelFileName}");

            var dData = data.Select(img => img.Cast<byte>().Select(d => d / 255.0));


            var MnistData = new List<TrainingData>();
            var MnistTest = new List<TrainingData>();

            int i = 0;
            foreach (var d in dData)
            {
                if (i % 2 == 0)
                {
                    MnistData.Add(new TrainingData(d.ToList(), CreateOneHotVector(10, label[i])));
                }
                else
                {
                    MnistData.Add(new TrainingData(d.ToList(), CreateOneHotVector(10, label[i])));
                }

                if (i++ > numSample)
                {
                    break;
                }
            }

            return (MnistData, MnistTest);

        }

        private static double[] CreateOneHotVector(int maxVal, int n)
        {
            var result = new double[maxVal];
            result[n] = 1.0;
            return result;
        }


        public static void PrintMnist(double[] data)
        {
            var width = 28;
            var height = 28;
            for (var i = 0; i < width; i++)
            {
                for (var j = 0; j < height; j++)
                {
                    Console.Write(Math.Round(data[i * width + j]) + " ");
                }
                Console.WriteLine();
            }
            Console.WriteLine(@"


");
        }

    }
}
