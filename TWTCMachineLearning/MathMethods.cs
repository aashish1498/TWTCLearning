namespace TWTCMachineLearning
{
    public static class MathMethods
    {
        public static double[,] MatrixMultiply(double[,] matrixA, double[,] matrixB)
        {
            double[,] newMatrix = new double[matrixA.GetLength(0),matrixB.GetLength(1)];
            for (int j = 0; j < matrixA.GetLength(0); j++)
            {
                for (int k = 0; k < matrixB.GetLength(1); k++)
                {
                    for (int i = 0; i < matrixA.GetLength(1); i++)
                    {
                        newMatrix[j, k] += matrixA[j, i] * matrixB[i, k];
                    }
                }
            }
            return newMatrix;
        }

        public static double[] MatrixMultiply(double[,] matrixA, double[] matrixB)
        {
            double[,] newMatrixB = new double[matrixB.Length,1];
            double[] newMatrix = new double[matrixA.GetLength(0)];

            for (int i = 0; i < matrixB.Length; i++)
            {
                newMatrixB[i, 0] = matrixB[i];
            }

            var newMatrixMulti = MatrixMultiply(matrixA, newMatrixB);

            for (int j = 0; j < newMatrix.Length; j++)
            {
                newMatrix[j] = newMatrixMulti[j, 0];
            }

            return newMatrix;
        }

        public static double[] MatrixAdd(double[] matrixA, double[] matrixB)
        {
            double[] newMatrix = new double[matrixA.Length];
            for (int i = 0; i < matrixA.Length; i++)
            {
                newMatrix[i] = matrixA[i] + matrixB[i];
            }

            return newMatrix;
        }
    }
}
