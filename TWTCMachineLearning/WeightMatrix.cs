using System;
using System.Collections.Generic;
using System.Text;

namespace TWTCMachineLearning
{
    public class WeightMatrix
    {
        public double[,] Values;

        public WeightMatrix(int dimension1, int dimension2)
        {
            Values = new double[dimension1, dimension2];
        }

        public bool Populate(double[,] valuesToCopy)
        {
            //if (valuesToCopy.Length != Values.Length) return false;
            int iLength = Values.GetLength(0);
            int jLength = Values.GetLength(1);

            for (int i = 0; i < iLength; i++)
            {
                for (int j = 0; j < jLength; j++)
                {
                    Values[i, j] = valuesToCopy[i, j];
                }
            }
            return true;
        }

        public void Randomise()
        {
            var randomMaster = new Random();
            for (int i = 0; i < Values.GetLength(0); i++)
            {
                for (int j = 0; j < Values.GetLength(1); j++)
                {
                    Values[i, j] = randomMaster.NextDouble() - 0.5;
                }
            }
        }
    }
}
