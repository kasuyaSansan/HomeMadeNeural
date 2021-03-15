using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace NeuralLib
{
    public class NeuralImageSingle
    {
        public float[][,] data;
        public int width;
        public int height;
        public int plane;

        public NeuralImageSingle(int plane, int width, int height)
        {
            this.width = width;
            this.height = height;
            this.plane = plane;
            data = new float[plane][,];
            for (var i = 0; i < plane; i++)
            {
                data[i] = new float[height, width];
            }
        }

        public NeuralImageSingle(double[][,] data)
        {
            this.plane = data.Length;
            this.height = data[0].GetLength(0);
            this.width = data[0].GetLength(1);
            this.data = new float[this.plane][,];

            for (int i = 0; i < plane; i++)
            {
                this.data[i] = new float[this.height, this.width];
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        this.data[i][y, x] = (float)data[i][x, y];
                    }
                }
            }
        }

        public NeuralImageSingle(float[][,] data)
        {
            this.plane = data.Length;
            this.height = data[0].GetLength(0);
            this.width = data[0].GetLength(1);
            this.data = new float[this.plane][,];

            for (int i = 0; i < plane; i++)
            {
                this.data[i] = new float[this.height, this.width];
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        this.data[i][y, x] = data[i][y, x];
                    }
                }
            }
        }

        public NeuralImageSingle(NeuralImage neuralImage)
        {
            height = neuralImage.height;
            width = neuralImage.width;
            plane = neuralImage.plane;
            var arrayData = new float[plane][,];
            for (int p = 0; p < plane; p++)
            {
                arrayData[p] = new float[height, width];
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        arrayData[p][i, j] = (float)neuralImage.data[p][i, j];
                    }
                }
            }

            this.data = arrayData;
        }

        public NeuralImageSingle(List<List<double>> data)
        {
            plane = 1;
            height = data.Count;
            width = data[0].Count;
            var arrayData = new float[0][,];
            arrayData[0] = new float[height, width];

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    arrayData[0][i, j] = (float)data[i][j];
                }
            }

            this.data = arrayData;
        }


        public NeuralImageSingle(List<List<double[]>> data)
        {

            height = data.Count;
            width = data[0].Count;
            plane = data[0][0].Length;
            var arrayData = new float[0][,];
            arrayData[0] = new float[height, width];
            for (int p = 0; p < plane; p++)
            {
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        arrayData[p][i, j] = (float)data[i][j][p];
                    }
                }
            }

            this.data = arrayData;
        }

        public NeuralImageSingle(double[] dataArray, int width, int height)
        {
            this.width = width;
            this.height = height;

            plane = 1;
            data = new float[1][,];
            data[0] = new float[height, width];
            for (int i = 0; i < dataArray.Length; i++)
            {
                var y = i / width;
                if (y >= height)
                {
                    break;
                }
                data[0][y, i - y * width] = (float)dataArray[i];
            }
        }

        public NeuralImageSingle(double[][] dataArray, int width, int height)
        {
            this.width = width;
            this.height = height;

            plane = dataArray[0].Length;
            data = new float[plane][,];

            for (int p = 0; p < plane; p++)
            {
                data[p] = new float[height, width];
                for (int i = 0; i < dataArray.Length; i++)
                {
                    var y = i / width;
                    if (y >= height)
                    {
                        break;
                    }

                    data[p][y, i - y * width] = (float)dataArray[i][p];
                }
            }
        }

        public NeuralImageSingle AddPaddingImage(int paddingLength)
        {
            var newImg = new NeuralImageSingle(this.plane, this.width + 2 * paddingLength, this.height + 2 * paddingLength);
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


        public NeuralImageSingle MatrixAppry(float[,][] filters, int length, int stride, int nextPlane)
        {
            var result = new float[nextPlane][,];
            for (int i = 0; i < nextPlane; i++)
            {
                var filterd = new float[plane][,];
                for (int j = 0; j < plane; j++)
                {
                    filterd[j] = ApplyFilterOne(filters[j, i], length, stride, j);
                }

                result[i] = SumAll(filterd);
            }
            return new NeuralImageSingle(result);
        }


        public NeuralImageSingle MatrixAppry3x3(Vector3[,][] filters, int nextPlane)
        {

            var result = new float[nextPlane][,];
            for (int i = 0; i < nextPlane; i++)
            {
                var filterd = new float[plane][,];
                for (int j = 0; j < plane; j++)
                {
                    filterd[j] = ApplyFilterOneV3x3(filters[j, i], j);
                }

                result[i] = SumAll(filterd);
            }
            return new NeuralImageSingle(result);
        }


        private static float[,] SumAll(float[][,] imgs)
        {
            var imgPlane = imgs.Length;
            var imgHeight = imgs[0].GetLength(0);
            var imgWidth = imgs[0].GetLength(1);

            var result = new float[imgHeight, imgWidth];
            for (int p = 0; p < imgPlane; p++)
            {
                for (int i = 0; i < imgHeight; i++)
                {
                    for (int j = 0; j < imgWidth; j++)
                    {
                        result[i, j] += imgs[p][i, j];
                    }
                }
            }

            return result;
        }


        public NeuralImageSingle ApplyFilters(float[][] filters, int length, int stride)
        {

            var numFilter = filters.Length;
            var outputImages = new List<NeuralImageSingle>();
            for (var i = 0; i < numFilter; i++)
            {
                var filtered = ApplyFilter(filters[i], length, stride);
                outputImages.Add(filtered);

            }
            return new NeuralImageSingle(outputImages.SelectMany(img => img.data).ToArray());
        }

        public (NeuralImageSingle, Dictionary<(int i, int ypos, int xpos), (int y, int x)>) ApplyMax(int filterSize, int stride)
        {
            var halfLength = (filterSize - 1) / 2;

            var stateData = new float[plane][,];
            var outputData = new float[plane][,];
            var newWidth = (int)Math.Ceiling((width - filterSize + stride) / (double)stride);
            var newHeight = (int)Math.Ceiling((height - filterSize + stride) / (double)stride);
            var argMaxMap = new Dictionary<(int, int, int), (int, int)>();

            for (var i = 0; i < plane; i++)
            {
                stateData[i] = new float[newHeight, newWidth];
                outputData[i] = new float[newHeight, newWidth];
                for (var ypos = 0; ypos < newHeight; ypos++)
                {
                    var centerY = ypos * stride + halfLength;

                    for (var xpos = 0; xpos < newWidth; xpos++)
                    {
                        outputData[i][ypos, xpos] = float.MinValue;
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
            return (new NeuralImageSingle(outputData), argMaxMap);
        }

        public float[,] ApplyFilterOneV3x3(Vector3[] filter, int planeIndex)
        {
            var halfLength = 1;

            var newWidth = width - 2;
            var newHeight = height - 2;
            var imgArea = new Vector3[3];
            var stateData = new float[newHeight, newWidth];
            unsafe
            {
                var v1 = new Vector3();
                var v2 = new Vector3();
                var v3 = new Vector3();

                fixed (float* dataPlane = &data[planeIndex][0, 0])
                    for (var ypos = 0; ypos < newHeight; ypos++)
                    {
                        var centerY = ypos + halfLength;

                        for (var xpos = 0; xpos < newWidth; xpos++)
                        {
                            var sy = centerY - halfLength;
                            var sx = xpos;
                            var ptr = sy * width + sx;
                            v1.X = dataPlane[ptr++];
                            v1.Y = dataPlane[ptr++];
                            v1.Z = dataPlane[ptr];
                            ptr = (sy + 1) * width + sx;
                            v2.X = dataPlane[ptr++];
                            v2.Y = dataPlane[ptr++];
                            v2.Z = dataPlane[ptr];
                            ptr = (sy + 2) * width + sx;
                            v3.X = dataPlane[ptr++];
                            v3.Y = dataPlane[ptr++];
                            v3.Z = dataPlane[ptr];

                            stateData[ypos, xpos] = Vector3.Dot(v1, filter[0])
                            + Vector3.Dot(v2, filter[1])
                            + Vector3.Dot(v3, filter[2]);
                        }
                    }
            }

            return stateData;
        }

        public float[,] ApplyFilterOne(float[] filter, int length, int stride, int planeIndex)
        {
            var halfLength = (length - 1) / 2;
            var newWidth = (int)Math.Floor((width - length + stride) / (double)stride);
            var newHeight = (int)Math.Floor((height - length + stride) / (double)stride);
            var filterPix = length * length;
            var imgArea = new float[filterPix];
            var stateData = new float[newHeight, newWidth];
            var dataPlane = data[planeIndex];

            unsafe
            {
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

                        var sum = 0.0f;
                        for (var idx = 0; idx < filterPix; idx++)
                            sum += imgArea[idx] * filter[idx];
                        stateData[ypos, xpos] = sum;
                    }
                }
            }
            return stateData;
        }

        public NeuralImageSingle ApplyFilter(float[] filter, int length, int stride)
        {
            var halfLength = (length - 1) / 2;
            var stateData = new float[plane][,];
            var newWidth = (int)Math.Floor((width - length + stride) / (double)stride);
            var newHeight = (int)Math.Floor((height - length + stride) / (double)stride);
            var filterPix = length * length;
            var imgArea = new float[filterPix];

            for (var i = 0; i < plane; i++)
            {
                stateData[i] = new float[newHeight, newWidth];
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

                        var sum = 0.0f;
                        for (var idx = 0; idx < filterPix; idx++)
                            sum += imgArea[idx] * filter[idx];
                        stateData[i][ypos, xpos] = sum;
                    }
                }
            }
            return new NeuralImageSingle(stateData);
        }
    }
}
