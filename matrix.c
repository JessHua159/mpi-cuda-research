#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Multiplies matrices mat1 and mat2 in serial and stores the result in matrix res
// res[i][j] = summed inner product of each element of mat1[i][k] with mat2[k][j]
void multiplyMatrix(float **mat1, float **mat2, float **res, int N)
{
   int i, j, k;
   for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
         res[i][j] = 0.0f;
         for (k = 0; k < N; k++)
            res[i][j] += mat1[i][k] * mat2[k][j];
      }
   }
}

// Ouputs matrix mat in 2d format
void displayMatrix(float **mat, int N)
{
   int i, j;
   for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++)
         printf("%.0f ", mat[i][j]);
      printf("\n");
   }
}

#define OUTPUT_MATRIX

int main(int argc, char **argv)
{

   int i, j, k, l;

   if (argc != 2) {
      printf("HighLife requires 1 argument\n");
      exit(-1);
   }

   int matrixSize = atoi(argv[1]);

   float **a, **b, **c;
   b = (float **)calloc(matrixSize, sizeof(float *));

   a = (float **)calloc(matrixSize, sizeof(float *));
   c = (float **)calloc(matrixSize, sizeof(float *));

   for (i = 0; i < matrixSize; i++) {
      a[i] = (float *)malloc(matrixSize * sizeof(float));
      b[i] = (float *)malloc(matrixSize * sizeof(float));
      c[i] = (float *)malloc(matrixSize * sizeof(float));
   }

   float n = 0.0f;
   for (i = 0; i < matrixSize; i++) {
      for (j = 0; j < matrixSize; j++) {
         a[i][j] = n++;
         b[i][j] = n++;
      }
   }

   // begin and end for measuring time of matrix multiply computation only
   clock_t begin = clock();
   multiplyMatrix(a, b, c, matrixSize);
   clock_t end = clock();
   #ifdef OUTPUT_MATRIX
      displayMatrix(a, matrixSize);
      displayMatrix(c, matrixSize);
   #endif

   for (i = 0; i < matrixSize; i++) {
      free(a[i]);
      free(b[i]);
      free(c[i]);
   }

   free(a);
   free(b);
   free(c);

   double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

   // time of matrix multiply computation only
   printf("Time taken for matrix multiplication: %f seconds\n", time_spent);

   return 0;
}
