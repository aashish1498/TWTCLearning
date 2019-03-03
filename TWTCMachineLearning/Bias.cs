using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace TWTCMachineLearning
{
    public class Bias
    {
        public double[] Values;

        public Bias(int size)
        {
            Values = new double[size];
        }

        public bool Populate(double[] valuesToCopy)
        {
            valuesToCopy.CopyTo(Values, 0);
            return true;
        }

        public void Randomise()
        {
            for (int i = 0; i < Values.Length; i++)
            {
                var randomMaster = new Random();
                Values[i] = randomMaster.NextDouble() - 0.5;
            }
        }
    }
}
