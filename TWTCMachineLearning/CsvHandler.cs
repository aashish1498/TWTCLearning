using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace TWTCMachineLearning
{
    public static class CsvHandler
    {
        private const char Delimeter = ',';
        public static double[] GetBias(string fileLocation)
        {
            var biasFull = File.ReadAllLines(fileLocation);
            double[] bias = new double[biasFull.Length];
            for (int i = 0; i < biasFull.Length; i++)
            {
                double.TryParse(biasFull[i], out var biasNumber);
                bias[i] = biasNumber;
            }

            return bias;
        }

        public static double[,] GetWeights(string fileLocation)
        {
            var weightsFull = File.ReadAllLines(fileLocation);
            var firstRow = weightsFull[0].Split(Delimeter);
            var weights = new double[weightsFull.Length,firstRow.Length];

            for (int i = 0; i < weightsFull.Length; i++) // Go through each row
            {
                var row = weightsFull[i].Split(Delimeter);
                for (int j = 0; j < row.Length; j++)
                {
                    double.TryParse(row[j], out var weightNumber);
                    weights[i, j] = weightNumber;
                }
            }

            return weights;
        }

        public static void SaveVector(double[] vector, string fileLocation)
        {
            var csv = new StringBuilder();
            foreach (var v in vector)
            {
                csv.Append(v + Environment.NewLine);
            }
            File.WriteAllText(fileLocation,csv.ToString());
        }

        public static void SaveWeights(double[,] weights, string fileLocation)
        {
            int iLength = weights.GetLength(0);
            int jLength = weights.GetLength(1);

            var csv = new StringBuilder();
            for (int i = 0; i < iLength; i++)
            {
                for (int j = 0; j < jLength; j++)
                {
                    csv.Append(weights[i, j] + Delimeter.ToString());
                }

                csv.Append(Environment.NewLine);
            }

            File.WriteAllText(fileLocation, csv.ToString());
        }

        public static void SaveWeights(float[,] weights, string fileLocation)
        {
            int iLength = weights.GetLength(0);
            int jLength = weights.GetLength(1);

            var csv = new StringBuilder();
            for (int i = 0; i < iLength; i++)
            {
                for (int j = 0; j < jLength; j++)
                {
                    csv.Append(weights[i, j] + Delimeter.ToString());
                }

                csv.Append(Environment.NewLine);
            }

            File.WriteAllText(fileLocation, csv.ToString());
        }

        public static InputsAndOutputs GetInputsAndOutputs(int noOfInputs, int noOfOutputs, string fileLocation)
        {
            List<double[]> inputList = new List<double[]>();
            List<double[]> outputList = new List<double[]>();
            var rawFile = File.ReadAllLines(fileLocation);

            foreach (var column in rawFile)
            {
                int j = 0;
                double[] inputs = new double[noOfInputs];
                double[] outputs = new double[noOfOutputs];

                var row = column.Split(Delimeter);
                while(j < noOfInputs)
                {
                    double.TryParse(row[j], out var inputNumber);
                    inputs[j] = inputNumber;
                    j++;
                }

                for (int k = 0; k < noOfOutputs; k++)
                {
                    double.TryParse(row[j], out var outputNumber);
                    outputs[k] = outputNumber;
                    j++;
                }
                inputList.Add(inputs);
                outputList.Add(outputs);
            }
            return new InputsAndOutputs(inputList, outputList);
        }
    }
}
