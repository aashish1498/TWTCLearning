using System.IO;

namespace TWTCMachineLearning
{
    public class NeuralNetwork
    {
        public Layer[] MyLayers;
        public Connection[] Connections;
        private readonly int noOfLayers;
        private readonly string name;
        private readonly string directoryName;

        /// <summary>
        /// Create a new Neural Network.
        /// </summary>
        /// <param name="startInfo">Starting information for the neural network</param>
        public NeuralNetwork(NetworkStartInfo startInfo)
        {
            name = startInfo.NetworkName;
            directoryName = startInfo.SaveLocation + @"\" + name + @"\";
            Directory.CreateDirectory(directoryName);
            noOfLayers = startInfo.LayerDetails.Length;
            MyLayers = new Layer[noOfLayers];
            for (int i = 0; i < noOfLayers; i++)
            {
                MyLayers[i] = new Layer(startInfo.LayerDetails[i]);
            }
            // Set up connections
            Connections = new Connection[noOfLayers - 1];
            SetUpConnections(Connections, startInfo.LearningRate, startInfo.ActivationMethod);
        }

        private void SetUpConnections(Connection[] connectionsArray, double learningRate, int activationMethod)
        {
            for (int i = 1; i < noOfLayers; i++)
            {
                var weights = new WeightMatrix(MyLayers[i].Values.Length, MyLayers[i - 1].Values.Length);
                var bias = new Bias(MyLayers[i].Values.Length);
                string weightsFilePath = directoryName + @"Weights\" + $"{name}_Weights_{i - 1}_{i}.csv";
                string biasesFilePath = directoryName + @"Biases\" + $"{name}_Biases_{i - 1}_{i}.csv";
                weights.Randomise();
                bias.Randomise();
                if (File.Exists(weightsFilePath))
                {
                    weights.Populate(CsvHandler.GetWeights(weightsFilePath));
                }
                if (File.Exists(biasesFilePath))
                {
                    bias.Populate(CsvHandler.GetBias(biasesFilePath));
                }
                connectionsArray[i - 1] = new Connection(MyLayers[i - 1], MyLayers[i], weights, bias, learningRate, activationMethod);
            }
        }

        /// <summary>
        ///  Finds the output for given inputs 
        /// </summary>
        /// <param name="inputs">Array of inputs</param>
        /// <returns></returns>
        public double[] FindOutput(double[] inputs)
        {
            for (int i = 0; i < noOfLayers - 1; i++)
            {
                if (i == 0)
                {
                    MyLayers[i].Populate(inputs);
                }
                Connections[i].PrevLayer = MyLayers[i];
                MyLayers[i + 1] = Connections[i].CalculateNextLayer();
            }

            return MyLayers[noOfLayers - 1].Values;
        }

        /// <summary>
        /// Trains the network based on what the expected output should have been. It is assumed that the 'FindOutput' method has been called for the associated inputs.
        /// </summary>
        /// <param name="expected">Expected output result</param>
        public void Learn(double[] expected)
        {
            for (int i = Connections.Length - 1; i >= 0; i--)
            {
                if (i == Connections.Length - 1)
                {
                    Connections[i].BackPropOutput(expected);
                }
                else
                {
                    Connections[i].BackPropHidden(Connections[i + 1].Gamma, Connections[i + 1].Weights);
                }
            }
        }

        public void SaveWeightsAndBiases()
        {
            Directory.CreateDirectory(directoryName + @"\Weights\");
            Directory.CreateDirectory(directoryName + @"\Biases\");

            for (int i = 1; i < noOfLayers; i++)
            {
                string weightsFilePath = directoryName + @"\Weights\" + $"{name}_Weights_{i - 1}_{i}.csv";
                string biasesFilePath = directoryName + @"\Biases\" + $"{name}_Biases_{i - 1}_{i}.csv";
                CsvHandler.SaveWeights(Connections[i - 1].Weights.Values, weightsFilePath);
                CsvHandler.SaveVector(Connections[i - 1].Biases.Values, biasesFilePath);
            }
        }
    }
}
