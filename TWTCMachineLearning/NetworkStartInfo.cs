namespace TWTCMachineLearning
{
    public class NetworkStartInfo
    {
        public readonly string NetworkName;
        public readonly int[] LayerDetails;
        public readonly double LearningRate;
        public readonly string SaveLocation;
        public readonly int ActivationMethod;

        /// <summary>
        /// Starting information for the neural network.
        /// </summary>
        /// <param name="name">Name of the network (used for saving/retrieving weights and biases)</param>
        /// <param name="layers">Details of how many neurons should be in each layer</param>
        /// <param name="learningRate">Learning rate of the program</param>
        /// <param name="location">Folder where the weights and biases will be saved/retrieved</param>
        /// <param name="activationMethod">Which method should be used for activation. 0 for tanh, 1 for sigmoid, 2 for ReLU</param>
        public NetworkStartInfo(string name, int[] layers, double learningRate, string location, int activationMethod)
        {
            NetworkName = name;
            LayerDetails = layers;
            LearningRate = learningRate;
            SaveLocation = location;
            ActivationMethod = activationMethod;
        }
    }
}
