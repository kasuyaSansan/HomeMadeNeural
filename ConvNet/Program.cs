using System;
using System.Linq;
using NeuralLib;
using System.IO;

namespace ConvNet
{

    class LeNet : ConvolutionalNeuralNet
    {
        //LeNetの実装。継承してコンストラクタでlayersにlayerを追加するとニューラルネットを作れる
        //継承せずに作ってlayersにAddしていっても作れる。
        public LeNet()
        {
            var learnRate = 0.01;
            layers.Add(new ConvolutionalLayer(5, 5, 1, 28, 28, 1, learnRate, true));
            layers.Add(new MaxPoolingLayer(3, 2, layers.Last()));
            layers.Add(new ConvolutionalLayer(3, 5, 1,layers.Last(), learnRate));
            layers.Add(new MaxPoolingLayer(3, 2, layers.Last()));
            layers.Add(new FullyConnectedLayer(120, layers.Last(), learnRate, FUNC_TYPE.Sigmoid));
            layers.Add(new FullyConnectedLayer(84, layers.Last(), learnRate, FUNC_TYPE.Sigmoid));
            layers.Add(new FullyConnectedLayer(10, layers.Last(), learnRate, FUNC_TYPE.Sigmoid));
        }
    }


    class Program
    {
        static void Main(string[] args)
        {
            var currentPath = Directory.GetCurrentDirectory();
            var mnistPath =currentPath + "/../../../../../NeuralLib/kuzushiji_MNIST";
            var data =MnistLoader.ReadImageData(4000, Path.Combine(mnistPath, "train-images-idx3-ubyte.gz"), Path.Combine(mnistPath, "train-labels-idx1-ubyte.gz"));

            var leNet = new LeNet();
            leNet.SetSample(data);
            leNet.Learn(200);
            var test = MnistLoader.ReadImageData(1000, Path.Combine(mnistPath, "t10k-images-idx3-ubyte.gz"), Path.Combine(mnistPath, "t10k-labels-idx1-ubyte.gz"));
            var successCount = 0.0;
            var totalCount = 0.0;
            foreach (var testData in test)
            {
                var result = leNet.InputImage(testData.Data);

                var resultVec = new double[10];
                var answerVec = new double[10];
                for(int i=0; i < 10; i++)
                {
                    resultVec[i] = result.data[0][0, i];
                    answerVec[i] = testData.Answer.data[0][0, i];
                }

                var argMax = resultVec.Select((d, i) => (d, i)).OrderByDescending(d => d.d).First().i;
                var answer = answerVec.Select((d, i) => (d, i)).OrderByDescending(d => d.d).First().i;
                if (argMax == answer)
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
