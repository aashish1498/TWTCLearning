using System.Collections.Generic;

namespace TWTCMachineLearning
{
    public class InputsAndOutputs
    {
        public InputsAndOutputs(List<double[]> inputs, List<double[]> outputs)
        {
            InputsList = inputs;
            OutputsList = outputs;
        }
        public List<double[]> InputsList;
        public List<double[]> OutputsList;
    }
}
