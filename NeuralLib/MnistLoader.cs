using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
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
        public static List<TrainingData>  ReadData(int numSample = 2000, string MnistFilePath = null, string MnistLabelFilePath = null)
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
                if(MnistFilePath == null || MnistLabelFilePath == null)
                    Console.WriteLine($"ReadDataは引数で指定がない場合、{autoFileName}と{autoFileName}を自動的に参照します。" +
                        $"ここに同名でファイルを置くか、ファイル名を指定してください\n" + e);
                else
                {
                    Console.WriteLine($"{MnistFilePath}、および、{MnistLabelFilePath}が指定され、\n{Path.GetFullPath(useFileName)}、および、{Path.GetFullPath(useLabelFileName)}を" +
                        $"読み込もうとしましたが、読み込むことができませんでした。上記のパスにファイルが存在するかまたはアクセス権限を確認ください");
                }
                throw;
            }

            var MnistData = new List<TrainingData>();

            int i = 0;
            foreach (var d in dData)
            {
                MnistData.Add(new TrainingData(d.ToList(), NeuralFunc.CreateOneHotVector(10, label[i])));
                if (i++ > numSample)
                {
                    break;
                }
            }

            return MnistData;
        }


        /// <summary>
        /// Mnistを画像形式(NeuralImage)で読み込みます
        /// </summary>
        /// <param name="numSample"></param>
        /// <param name="MnistFilePath"></param>
        /// <param name="MnistLabelFilePath"></param>
        /// <returns></returns>
        public static List<TrainingImageData> ReadImageData(int numSample = 2000, string MnistFilePath = null, string MnistLabelFilePath = null)
        {

            var data = ReadData(numSample, MnistFilePath, MnistLabelFilePath); 
            var MnistData = new List<TrainingImageData>();

            int i = 0;
            foreach (var d in data)
            {
                MnistData.Add(new TrainingImageData(new NeuralImage(d.Data.ToArray(), 28, 28), new NeuralImage(d.Answer.ToArray(), 10, 1)));
                if (i++ > numSample)
                {
                    break;
                }
            }

            return MnistData;
        }

        /// <summary>
        /// ministDataを可視化
        /// </summary>
        /// <param name="mnistData"></param>
        public static void PrintMnist(List<double> mnistData)
        {
            for(int y =0; y < 28; y++)
            {
                for(int x=0; x < 28; x++)
                {
                    Console.Write((int)Math.Round(mnistData[y * 28 + x]) + " ");
                }
                Console.WriteLine();
            }
        }
    }
}
