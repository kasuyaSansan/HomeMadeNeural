using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralLib
{
    /// <summary>
    /// 学習データ用のクラス。入力も出力もdoubleのリストとする
    /// </summary>
    public class TrainingData
    {
        public List<double> Answer;
        public List<double> Data;

        public TrainingData(IReadOnlyList<double> data, IReadOnlyList<double> answer)
        {
            Answer = answer.ToList();
            Data = data.ToList();
        }
    }
}
