using System;
using static TWTCMachineLearning.MathMethods;

namespace TWTCMachineLearning
{
    public class Connection
    {
        private int _activationMethod;
        private double _learningRate;
        private WeightMatrix _deltaWeights;
        private Bias _deltaBias;

        public Layer NextLayer;
        public Layer PrevLayer;
        public WeightMatrix Weights;
        public Bias Biases;

        public double[] Error;
        public double[] Gamma;

        public Connection(Layer previousLayer, Layer nextLayer, WeightMatrix weightMatrix, Bias bias, double learningRate, int activationMethod)
        {
            _activationMethod = activationMethod;
            _learningRate = learningRate;
            PrevLayer = previousLayer;
            NextLayer = nextLayer;
            Weights = weightMatrix;
            Biases = bias;
            _deltaWeights = new WeightMatrix(nextLayer.Values.Length, previousLayer.Values.Length);
            _deltaBias = new Bias(nextLayer.Values.Length);
        }

        public Layer CalculateNextLayer()
        {
            var newLayerValues = MatrixAdd(MatrixMultiply(Weights.Values, PrevLayer.Values), Biases.Values);
            NextLayer.Values = Activation(newLayerValues);
            return NextLayer;
        }

        public void BackPropOutput(double[] expected)
        {
            Error = new double[expected.Length];
            Gamma = new double[expected.Length];
            // Find gamma
            for (int i = 0; i < expected.Length; i++)
            {
                Error[i] = NextLayer.Values[i] - expected[i];
            }
            for (int i = 0; i < expected.Length; i++)
            {
                Gamma[i] = Error[i] * SingleActivationPrime(NextLayer.Values[i]);
            }

            // Find delta weights and biases
            for (int i = 0; i < expected.Length; i++)
            {
                for (int j = 0; j < PrevLayer.Values.Length; j++)
                {
                    _deltaWeights.Values[i, j] = Gamma[i] * PrevLayer.Values[j];
                }
            }

            for (int i = 0; i < NextLayer.Values.Length; i++)
            {
                _deltaBias.Values[i] = Gamma[i];
            }

            UpdateWeightsAndBiases();
        }

        public void BackPropHidden(double[] gammaForward, WeightMatrix nextWeights)
        {

            double[,] weightsForward = nextWeights.Values;
            Gamma = new double[NextLayer.Values.Length];

            // Calculate new gamma using gamma sums of the forward layer
            for (int i = 0; i < NextLayer.Values.Length; i++)
            {
                Gamma[i] = 0;

                for (int j = 0; j < gammaForward.Length; j++)
                {
                    Gamma[i] += gammaForward[j] * weightsForward[j, i];
                }

                Gamma[i] *= SingleActivationPrime(NextLayer.Values[i]);
            }

            // Calculate delta weights
            for (int i = 0; i < NextLayer.Values.Length; i++)
            {
                for (int j = 0; j < PrevLayer.Values.Length; j++)
                {
                    _deltaWeights.Values[i, j] = Gamma[i] * PrevLayer.Values[j];
                }
            }

            for (int i = 0; i < NextLayer.Values.Length; i++)
            {
                _deltaBias.Values[i] = Gamma[i];
            }

            UpdateWeightsAndBiases();
        }

        private void UpdateWeightsAndBiases()
        {
            for (int i = 0; i < NextLayer.Values.Length; i++)
            {
                for (int j = 0; j < PrevLayer.Values.Length; j++)
                {
                    Weights.Values[i, j] -= _deltaWeights.Values[i, j] * _learningRate;
                }
            }

            for (int i = 0; i < NextLayer.Values.Length; i++)
            {
                Biases.Values[i] -= _deltaBias.Values[i] * _learningRate;
            }
        }

        public double[] Activation(double[] oldDoubles)
        {
            double[] newDoubles = new double[oldDoubles.Length];
            int i = 0;
            foreach (var d in oldDoubles)
            {
                newDoubles[i] = SingleActivation(d);
                i++;
            }

            return newDoubles;
        }

        private double SingleActivation(double x)
        {
            switch (_activationMethod)
            {
                case 0:
                    return Math.Tanh(x);
                case 1:
                    return 1 / (1 + Math.Exp(-x));
                default:
                    return Math.Max(x, 0);

            }
        }

        public double SingleActivationPrime(double x)
        {
            switch (_activationMethod)
            {
                case 0:
                    return 1 - (x * x);
                case 1:
                    return SingleActivation(x) * (1 - SingleActivation(x));
                default:
                    return x < 0 ? 0.01 * x : 1;
            }
        }
    }
}
