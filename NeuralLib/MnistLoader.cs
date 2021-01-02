using System;
using System.Collections.Generic;
using System.Linq;
using MNIST.IO;

namespace NeuralLib
{
    public class MnistLoader
    {
        private static string normalMnistFileName = "train-images-idx3-ubyte.gz";
        private static string normalMnistLabelName = "train-labels-idx1-ubyte.gz";


        /// <summary>
        /// MnistFilePath, MnistLabelFilePathを指定しないと、c:/data/minst/train-images-idx3-ubyte.gz, c:/data/minist/train-labels-idx1-ubyte.gz
        /// から自動的にファイルを読み込みます。
        /// </summary>
        /// <param name="numSample"></param>
        /// <param name="MnistFilePath"></param>
        /// <param name="MnistLabelFilePath"></param>
        /// <returns></returns>
        public static (List<TrainingData> MnistData, List<TrainingData> MnistTest) ReadData(int numSample = 2000, string MnistFilePath = null, string MnistLabelFilePath = null)
        {
            var autoFileName = $"c:/data/mnist/{normalMnistFileName}";
            var useFileName = autoFileName;
            var autoLabelFileName = $"c:/data/mnist/{normalMnistLabelName}";
            var useLabelFileName = autoLabelFileName;

            if (!string.IsNullOrEmpty(MnistFilePath))
            {
                useFileName = MnistFilePath;
            }
            if (!string.IsNullOrEmpty(MnistLabelFilePath))
            {
                useLabelFileName = MnistLabelFilePath;
            }

            IEnumerable<IEnumerable<double>> dData;
            byte[] label;
            try
            {
                var data = FileReaderMNIST.LoadImages(useFileName);
                label = FileReaderMNIST.LoadLabel(useLabelFileName);
                dData = data.Select(img => img.Cast<byte>().Select(d => d / 255.0));
            }
            catch (Exception e)
            {
                Console.WriteLine($"ReadDataは引数で指定がない場合、{autoFileName}と{autoFileName}を自動的に参照します。" +
                    $"ここに同名でファイルを置くか、ファイル名を指定してください\n" + e);
                throw;
            }


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
    }
}
