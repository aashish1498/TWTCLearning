namespace TWTCMachineLearning
{
    public class Layer
    {
        public double[] Values;
        public Layer(int size)
        {
            Values = new double[size];
        }

        public bool Populate(double[] valuesToCopy)
        {
            valuesToCopy.CopyTo(Values, 0);
            return true;
        }
    }
}
