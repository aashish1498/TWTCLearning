using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TWTCMachineLearning;


namespace AI_Test
{
    [TestClass]
    public class UnitTest1
    {
        private string testTrainingData1 = @"C:\Users\Roopal\Documents\Aashish\Shellshock\Training Data Archive\TrainingDataZeroNormalised.csv";
        private string testTrainingData2 = @"C:\Users\Roopal\Documents\Aashish\Shellshock\Training Data Archive\TrainingDataOneNormalised.csv";
        private string testTrainingData3 = @"C:\Users\Roopal\Documents\Aashish\Shellshock\Training Data Archive\TrainingDataTwoNormalised.csv";
        private string testTrainingData4 = @"C:\Users\Roopal\Documents\Aashish\Shellshock\Training Data Archive\TrainingDataThreeNormalised.csv";

        [TestMethod]
        public void TestExistingTrainingData()
        {
            List<InputsAndOutputs> listOfIO = new List<InputsAndOutputs>();
            InputsAndOutputs inputsAndOutputs1 = CsvHandler.GetInputsAndOutputs(19, 2, testTrainingData1);
            InputsAndOutputs inputsAndOutputs2 = CsvHandler.GetInputsAndOutputs(19, 2, testTrainingData2);
            InputsAndOutputs inputsAndOutputs3 = CsvHandler.GetInputsAndOutputs(19, 2, testTrainingData3);
            InputsAndOutputs inputsAndOutputs4 = CsvHandler.GetInputsAndOutputs(19, 2, testTrainingData4);
            listOfIO.Add(inputsAndOutputs1);
            listOfIO.Add(inputsAndOutputs2);
            listOfIO.Add(inputsAndOutputs3);
            listOfIO.Add(inputsAndOutputs4);

            //InputsAndOutputs io = CsvHandler.GetInputsAndOutputs(19, 2, testTrainingData1);
            NetworkStartInfo startInfo = new NetworkStartInfo(
                name: "ArtificialShellshock_Final",
                layers: new[] { 19, 25, 25, 2 },
                learningRate: 0.03,
                location: @"C:\Users\Roopal\Documents\Aashish\Shellshock",
                activationMethod: 0);
            NeuralNetwork aiMain = new NeuralNetwork(startInfo);
            List<double> averageCost = new List<double>();
            foreach (var io in listOfIO)
            {
                for (int i = 0; i < 5001; i++)
                {
                    double[] costs = new double[io.InputsList.Count];
                    for (int j = 0; j < io.InputsList.Count; j++)
                    {
                        aiMain.FindOutput(io.InputsList[j]);
                        costs[j] = (Math.Abs(aiMain.MyLayers[3].Values[0] - io.OutputsList[j][0]) +
                                            Math.Abs(aiMain.MyLayers[3].Values[1] - io.OutputsList[j][1])) / 2;

                        aiMain.Learn(io.OutputsList[j]);
                    }

                    if (i % 1000 == 0)
                    {
                        aiMain.SaveWeightsAndBiases();
                    }

                    if (i % 100 == 0)
                    {
                        averageCost.Add(costs.Average());
                    }
                }
            }
            double[] avgCost = new double[averageCost.Count];
            averageCost.CopyTo(avgCost, 0);
            CsvHandler.SaveVector(avgCost,
                @"C:\Users\Roopal\Documents\Aashish\Shellshock\ArtificialShellshock_Final" + @"\AverageCost.csv");
            var actualOutput = aiMain.FindOutput(listOfIO[3].InputsList[0]);
            Assert.AreEqual(actualOutput[0], listOfIO[0].OutputsList[0][0], 0.1);
        }

        [TestMethod]
        public void TestLearning()
        {
            NetworkStartInfo startInfo = new NetworkStartInfo(
                name: "myTestTwo",
                layers: new[] { 3, 25, 25, 1 },
                learningRate: 0.03,
                location: @"C:\Users\Roopal\Documents\Aashish\Shellshock",
                activationMethod: 0);
            NeuralNetwork _aiMain = new NeuralNetwork(startInfo);
            for (int i = 0; i < 5000; i++)
            {
                _aiMain.FindOutput(new double[] { 0, 0, 0 });
                _aiMain.Learn(new double[] { 0.91 });

                _aiMain.FindOutput(new double[] { 0, 0, 1 });
                _aiMain.Learn(new double[] { 1 });

                _aiMain.FindOutput(new double[] { 0, 1, 0 });
                _aiMain.Learn(new double[] { 0.14 });

                _aiMain.FindOutput(new double[] { 0, 1, 1 });
                _aiMain.Learn(new double[] { 0 });

                _aiMain.FindOutput(new double[] { 1, 0, 0 });
                _aiMain.Learn(new double[] { -0.21 });

                _aiMain.FindOutput(new double[] { 1, 0, 1 });
                _aiMain.Learn(new double[] { 0 });

                _aiMain.FindOutput(new double[] { 1, 1, 0 });
                _aiMain.Learn(new double[] { 0.63 });

                _aiMain.FindOutput(new double[] { 1, 1, 1 });
                _aiMain.Learn(new double[] { -0.8 });
            }


            var actualOutputs = _aiMain.FindOutput(new double[] { 1, 1, 1 });
            Assert.AreEqual(-0.8, actualOutputs[0], 0.05);

            actualOutputs = _aiMain.FindOutput(new double[] { 1, 1, 0 });
            Assert.AreEqual(0.63, actualOutputs[0], 0.05);

            actualOutputs = _aiMain.FindOutput(new double[] { 1, 0, 0 });
            Assert.AreEqual(-0.21, actualOutputs[0], 0.05);

            actualOutputs = _aiMain.FindOutput(new double[] { 0, 0, 0 });
            Assert.AreEqual(0.91, actualOutputs[0], 0.05);

            actualOutputs = _aiMain.FindOutput(new double[] { 0, 1, 0 });
            Assert.AreEqual(0.14, actualOutputs[0], 0.05);

            _aiMain.SaveWeightsAndBiases();
        }

        [TestMethod]
        public void TestExistingWeights()
        {
            NetworkStartInfo startInfo = new NetworkStartInfo(
                name: "myTestTwo",
                layers: new[] { 3, 25, 25, 1 },
                learningRate: 0.03,
                location: @"C:\Users\Roopal\Documents\Aashish\Shellshock",
                activationMethod: 0);
            NeuralNetwork _aiMain = new NeuralNetwork(startInfo);
            for (int i = 0; i < 5000; i++)
            {
                _aiMain.FindOutput(new double[] { 0, 0, 0 });

                _aiMain.FindOutput(new double[] { 0, 0, 1 });

                _aiMain.FindOutput(new double[] { 0, 1, 0 });

                _aiMain.FindOutput(new double[] { 0, 1, 1 });

                _aiMain.FindOutput(new double[] { 1, 0, 0 });

                _aiMain.FindOutput(new double[] { 1, 0, 1 });

                _aiMain.FindOutput(new double[] { 1, 1, 0 });

                _aiMain.FindOutput(new double[] { 1, 1, 1 });
            }

            var actualOutputs = _aiMain.FindOutput(new double[] { 1, 1, 1 });
            Assert.AreEqual(-0.8, actualOutputs[0], 0.05);

            actualOutputs = _aiMain.FindOutput(new double[] { 1, 1, 0 });
            Assert.AreEqual(0.63, actualOutputs[0], 0.05);

            actualOutputs = _aiMain.FindOutput(new double[] { 1, 0, 0 });
            Assert.AreEqual(-0.21, actualOutputs[0], 0.05);

            actualOutputs = _aiMain.FindOutput(new double[] { 0, 0, 0 });
            Assert.AreEqual(0.91, actualOutputs[0], 0.05);

            actualOutputs = _aiMain.FindOutput(new double[] { 0, 1, 0 });
            Assert.AreEqual(0.14, actualOutputs[0], 0.05);
        }

        [TestMethod]
        public void TestMatrixMultiplication()
        {
            var matrixA = new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
            var matrixB = new double[] { 7, 8, 9 };

            var newMatrix = MathMethods.MatrixMultiply(matrixA, matrixB);

            Assert.AreEqual(122, newMatrix[1]);
        }

        [TestMethod]
        public void TestMatrixAddition()
        {
            var matrixA = new double[] { 1, 7, 3 };
            var matrixB = new double[] { 7, 18, 9 };

            var newMatrix = MathMethods.MatrixAdd(matrixA, matrixB);

            Assert.AreEqual(25, newMatrix[1]);
        }
    }
}