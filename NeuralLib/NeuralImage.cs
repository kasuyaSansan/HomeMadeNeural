using System;
using System.Collections.Generic;
using System.IO;
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


        public NeuralImage(NeuralImageSingle neuralImageS)
        {
            height = neuralImageS.height;
            width = neuralImageS.width;
            plane = neuralImageS.plane;
            var arrayData = new double[plane][,];

            for (int p = 0; p < plane; p++)
            {
                arrayData[p] = new double[height, width];
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        arrayData[p][i, j] = neuralImageS.data[p][i, j];
                    }
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

        public NeuralImage AddPaddingImage(int paddingLength)
        {
            var newImg = new NeuralImage(this.plane, this.width + 2 * paddingLength, this.height + 2 * paddingLength);
            for (int p = 0; p < plane; p++)
            {


                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        newImg.data[p][i + paddingLength, j + paddingLength] = data[p][i, j];
                    }
                }
            }

            return newImg;
        }

        public NeuralImageSingle AddPaddingImageS(int paddingLength)
        {
            var newImg = new NeuralImageSingle(this.plane, this.width + 2 * paddingLength, this.height + 2 * paddingLength);
            for (int p = 0; p < plane; p++)
            {
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        newImg.data[p][i + paddingLength, j + paddingLength] = (float)data[p][i, j];
                    }
                }
            }

            return newImg;
        }

        public NeuralImage MatrixAppry(double[,][] filters, int length, int stride, int nextPlane)
        {
            var result = new double[nextPlane][,];
            for (int i = 0; i < nextPlane; i++)
            {
                var sumFilter = ApplyFilterOne(filters[0, i], length, stride, 0);
                for (int j = 1; j < plane; j++)
                {
                    var filterd = ApplyFilterOne(filters[j, i], length, stride, j);
                    sumFilter = Sum(sumFilter, filterd);
                }

                result[i] = sumFilter;
            }
            return new NeuralImage(result);
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
                                    //if (argMaxMap.ContainsKey((i, ypos, xpos)))
                                    //{
                                    //    argMaxMap[(i, ypos, xpos)] = (y, x);
                                    //}
                                    //else
                                    //{
                                    //    argMaxMap.Add((i, ypos, xpos), (y, x));
                                    //}
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

        private static double[,] Sum(double[,] img1, double[,] img2)
        {
            var imgHeight = img1.GetLength(0);
            var imgWidth = img2.GetLength(1);
            var result = new double[imgHeight, imgWidth];
            for (int i = 0; i < imgHeight; i++)
            {
                for (int j = 0; j < imgWidth; j++)
                {
                    result[i, j] = img1[i, j] + img2[i, j];
                }
            }

            return result;
        }

        public double[,] ApplyFilterOne(double[] filter, int length, int stride, int planeIndex)
        {
            var halfLength = (length - 1) / 2;


            var newWidth = (int)Math.Floor((width - length + stride) / (double)stride);
            var newHeight = (int)Math.Floor((height - length + stride) / (double)stride);
            var filterPix = length * length;
            var imgArea = new double[filterPix];


            var stateData = new double[newHeight, newWidth];
            var dataPlane = data[planeIndex];


            for (var ypos = 0; ypos < newHeight; ypos++)
            {
                var centerY = ypos * stride + halfLength;

                for (var xpos = 0; xpos < newWidth; xpos++)
                {
                    var sy = centerY - halfLength;
                    var ey = sy + length;

                    // var centerX = xpos * stride + halfLength;
                    var sx = xpos * stride;
                    var ex = sx + length;
                    var m = 0;
                    for (var y = sy; y < ey; y++)
                        for (var x = sx; x < ex; x++)
                            imgArea[m++] = dataPlane[y, x];

                    var sum = 0.0;
                    for (var idx = 0; idx < filterPix; idx++)
                        sum += imgArea[idx] * filter[idx];
                    stateData[ypos, xpos] = sum;

                }
            }

            return stateData;
        }

        public NeuralImage ApplyFunction(ActiveFunction func, double param)
        {
            var result = new double[plane][,];
            for (int p = 0; p < plane; p++)
            {
                result[p] = new double[height, width];
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        result[p][i, j] = func(data[p][i, j], param);
                    }
                }
            }

            return new NeuralImage(result);
        }

        public NeuralImage ApplyGlobalAverage()
        {
            var n = width * height;
            var result = new NeuralImage(1, plane, 1);

            for (var i = 0; i < plane; i++)
            {
                for (var y = 0; y < height; y++)
                {
                    for (var x = 0; x < width; x++)
                    {
                        result.data[0][0, i] += data[i][y, x];
                    }
                }
                result.data[0][0, i] /= n;
            }

            return result;
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

                        // var centerX = xpos * stride + halfLength;
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
