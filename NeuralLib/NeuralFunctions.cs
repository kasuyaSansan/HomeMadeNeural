using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralLib
{
    public enum FUNC_TYPE
    {
        ReLU, Sigmoid, LeakyReLU
    }
    public delegate double ActiveFunction(double x, double param);
    public delegate double DifferentialActiveFunction(double x, double param);
    public delegate float ActiveFunctionf(float x, float param);
    public delegate float DifferentialActiveFunctionf(float x, float param);
    public static class NeuralFunc
    {
        public static double Sigmoid(double x, double a)
        {
            return (1 / (1 + Math.Exp(-a * x)));
        }

        public static double Dsigmoid(double x, double a)
        {
            var f = Sigmoid(x, a);
            return a * f * (1.0 - f);
        }

        public static double ReLU(double x, double a)
        {
            if (x > 0) return x;
            return 0;
        }

        public static double DReLU(double x, double a)
        {
            if (x > 0) return 1;
            return 0;
        }

        public static double LeakyReLU(double x, double a)
        {
            if (x > 0) return x;
            return a * x;
        }

        public static double DLeakyReLU(double x, double a)
        {
            if (x > 0) return 1;
            return a;
        }

        public static double[] SoftMax(double[] x)
        {
            var ex = x.Select(Math.Exp);
            var exSum = ex.Sum();
            return ex.Select(a => a / exSum).ToArray();
        }

        public static double[,] DSoftMax(double[] y)
        {
            var length = y.Length;
            var dydx = new double[length, length];
            for (var i = 0; i < length; i++)
            {
                for (var j = 0; j < length; j++)
                {
                    if (i == j)
                        dydx[i, j] = y[i] * (1 - y[i]);
                    else
                        dydx[i, j] = -y[i] * y[j];
                }
            }
            return dydx;
        }


        public static float Sigmoid(float x, float a)
        {
            return (float)(1 / (1 + Math.Exp(-a * x)));
        }

        public static float Dsigmoid(float x, float a)
        {
            var f = Sigmoid(x, a);
            return a * f * (1.0f - f);
        }

        public static float ReLU(float x, float a)
        {
            if (x > 0) return x;
            return 0;
        }

        public static float DReLU(float x, float a)
        {
            if (x > 0) return 1;
            return 0;
        }

        public static float LeakyReLU(float x, float a)
        {
            if (x > 0) return x;
            return a * x;
        }

        public static float DLeakyReLU(float x, float a)
        {
            if (x > 0) return 1;
            return a;
        }


        public static double[] CreateOneHotVector(int numClass, int oneIndex)
        {
            var result = new double[numClass];
            result[oneIndex] = 1.0;
            return result;
        }

        public static List<List<T>> SplitToBatch<T>(IEnumerable<T> sample, int n)
        {
            return sample.OrderBy(d => Guid.NewGuid())
                .Select((d, i) => (d, i / n))
                .GroupBy(d => d.Item2)
                .Select(g => g.Select<(T d, int), T>(x => x.Item1).ToList()).ToList();
        }

        public static List<T> GetNSample<T>(IEnumerable<T> sample, int n)
        {
            return SplitToBatch(sample, n)[0];
        }


        public static (int, int)[][] CreateConnectionMap(int length, int filterSize, int stride)
        {
            var connectionMap = new Dictionary<int, List<(int, int)>>();
            for (var i = 0; i < length; i++)
            {
                connectionMap.Add(i, new List<(int, int)>());
            }

            for (var i = 0; i + filterSize - 1 < length; i += stride)
            {
                for (var j = 0; j < filterSize; j++)
                {
                    connectionMap[i + j].Add((i / stride, j));
                }
            }

            var arrayConnectionMap = new (int, int)[length][];
            for (var i = 0; i < length; i++)
            {
                arrayConnectionMap[i] = connectionMap[i].ToArray();
            }
            return arrayConnectionMap;
        }

        private static Random rand = new Random();
        public static double BoxMuler(double mean, double stdDev)
        {

            double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                   Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal =
                mean + stdDev * randStdNormal;
            return randNormal;
        }
    }
}
