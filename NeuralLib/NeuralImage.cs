using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralLib
{
    public class NeuralImage
    {
        public double[][,] data;
        public int width;
        public int height;
        public int plane;


        public NeuralImage(int plane, int width, int height)
        {
            this.width = width;
            this.height = height;
            this.plane = plane;
            data = new double[plane][,];
            for (var i = 0; i < plane; i++)
            {
                data[i] = new double[height, width];
            }
        }

        public NeuralImage(double[][,] data)
        {
            this.plane = data.Length;
            this.height = data[0].GetLength(0);
            this.width = data[0].GetLength(1);
            this.data = data;
        }

        public NeuralImage(List<List<double>> data)
        {
            plane = 1;
            height = data.Count;
            width = data[0].Count;
            var arrayData = new double[0][,];
            arrayData[0] = new double[height, width];

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    arrayData[0][i, j] = data[i][j];
                }
            }

            this.data = arrayData;
        }

        public NeuralImage(double[] dataArray, int width, int height)
        {
            this.width = width;
            this.height = height;

            plane = 1;
            data = new double[1][,];
            data[0] = new double[height, width];
            for (int i = 0; i < dataArray.Length; i++)
            {
                var y = i / width;
                if (y >= height)
                {
                    break;
                }
                data[0][y, i - y * width] = dataArray[i];
            }
        }

        public NeuralImage(double[][] dataArray, int width, int height)
        {
            this.width = width;
            this.height = height;

            plane = dataArray[0].Length;
            data = new double[plane][,];

            for (int p = 0; p < plane; p++)
            {
                data[p] = new double[height, width];
                for (int i = 0; i < dataArray.Length; i++)
                {
                    var y = i / width;
                    if (y >= height)
                    {
                        break;
                    }

                    data[p][y, i - y * width] = dataArray[i][p];
                }
            }
        }

        public NeuralImage ApplyFilters(double[][] filters, int length, int stride)
        {
            var numFilter = filters.Length;
            var outputImages = new List<NeuralImage>();
            for (var i = 0; i < numFilter; i++)
            {
                var filtered = ApplyFilter(filters[i], length, stride);
                outputImages.Add(filtered);
            }
            return new NeuralImage(outputImages.SelectMany(img => img.data).ToArray());
        }

        public (NeuralImage, Dictionary<(int i, int ypos, int xpos), (int y, int x)>) ApplyMax(int filterSize, int stride)
        {
            var halfLength = (filterSize - 1) / 2;

            var stateData = new double[plane][,];
            var outputData = new double[plane][,];
            var newWidth = (int)Math.Ceiling((width - filterSize + stride) / (double)stride);
            var newHeight = (int)Math.Ceiling((height - filterSize + stride) / (double)stride);
            var argMaxMap = new Dictionary<(int, int, int), (int, int)>();

            for (var i = 0; i < plane; i++)
            {
                stateData[i] = new double[newHeight, newWidth];
                outputData[i] = new double[newHeight, newWidth];
                for (var ypos = 0; ypos < newHeight; ypos++)
                {
                    var centerY = ypos * stride + halfLength;

                    for (var xpos = 0; xpos < newWidth; xpos++)
                    {
                        outputData[i][ypos, xpos] = double.MinValue;
                        var centerX = xpos * stride + halfLength;
                        var y = centerY - halfLength;
                        var argMax = (0, 0);
                        for (var filterYpos = 0; filterYpos < filterSize; filterYpos++)
                        {
                            var x = centerX - halfLength;
                            if (y >= height)
                                continue;
                            for (var filterXpos = 0; filterXpos < filterSize; filterXpos++)
                            {
                                if (x >= width)
                                    continue;
                                if (data[i][y, x] > outputData[i][ypos, xpos])
                                {
                                    outputData[i][ypos, xpos] = data[i][y, x];
                                    argMax = (y, x);
                                }
                                x++;
                            }
                            y++;
                        }
                        argMaxMap.Add((i, ypos, xpos), argMax);
                    }
                }
            }
            return (new NeuralImage(outputData), argMaxMap);
        }

        public NeuralImage ApplyFilter(double[] filter, int length, int stride)
        {
            var halfLength = (length - 1) / 2;

            var stateData = new double[plane][,];
            var newWidth = (int)Math.Floor((width - length + stride) / (double)stride);
            var newHeight = (int)Math.Floor((height - length + stride) / (double)stride);
            var filterPix = length * length;
            var imgArea = new double[filterPix];

            for (var i = 0; i < plane; i++)
            {
                stateData[i] = new double[newHeight, newWidth];
                var dataPlane = data[i];

                for (var ypos = 0; ypos < newHeight; ypos++)
                {
                    var centerY = ypos * stride + halfLength;

                    for (var xpos = 0; xpos < newWidth; xpos++)
                    {
                        var sy = centerY - halfLength;
                        var ey = sy + length;

                        var sx = xpos * stride;
                        var ex = sx + length;
                        var m = 0;
                        for (var y = sy; y < ey; y++)
                            for (var x = sx; x < ex; x++)
                                imgArea[m++] = dataPlane[y, x];

                        var sum = 0.0;
                        for (var idx = 0; idx < filterPix; idx++)
                            sum += imgArea[idx] * filter[idx];
                        stateData[i][ypos, xpos] = sum;
                    }
                }
            }
            return new NeuralImage(stateData);
        }
    }
}