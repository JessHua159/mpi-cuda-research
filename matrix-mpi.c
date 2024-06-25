#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <stdint.h>

#ifndef CLOCKCYCLE_H
#define CLOCKCYCLE_H

// Retrieves clock cycle counter from system
uint64_t clock_now(void)
{
   unsigned int tbl, tbu0, tbu1;

   do {
      __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
      __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
      __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
   } while (tbu0 != tbu1);
   return (((uint64_t)tbu0) << 32) | tbl;
}
#endif

// Outputs matrix mat in 2d format
void displayMatrix(float *mat, int r, int c)
{
   int i, j;
   for (i = 0; i < r; i++) {
      for (j = 0; j < c; j++)
         printf("%.0f ", mat[j + i * c]);
      printf("\n");
   }
}

#define OUTPUT_MATRIX
#define IO
#define FILENAME "matrix_multiply_result.txt"

int main(int argc, char **argv)
{
   // begin_clock and end_clock to measure total time for matrix multiplication,
   // including message passing and synchronization overhead
   uint64_t begin_clock, end_clock;

   // Initialize the MPI environment
   MPI_Init(&argc, &argv);

   // Get the number of processes
   int numProcs;
   MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

   // Get the rank of the process
   int myRank;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

   if (argc != 2) {
      printf("HighLife requires 1 argument\n");
      exit(-1);
   }
   int matrixSize = atoi(argv[1]);
   if (matrixSize % numProcs != 0) {
      printf("Matrix size must be divisible by the number of processes\n");
      exit(-1);
   }
   int rowsPerProc = matrixSize / numProcs;
   size_t cellsPerProc = rowsPerProc * matrixSize;
   int i, j, k;
   float *a, *b, *c;
   double computation_time;
   b = calloc(matrixSize * matrixSize, sizeof(float));
   float *sentRow = calloc(cellsPerProc, sizeof(float));

   // begin_clock_mult and end_clock_mult to measure time of matrix multiply computation only
   uint64_t begin_clock_mult, end_clock_mult;

   if (myRank == 0) {
      a = calloc(matrixSize * matrixSize, sizeof(float));

      c = calloc(matrixSize * matrixSize, sizeof(float));

      float n = 0.0f;
      for (i = 0; i < matrixSize*matrixSize; i++) {
         a[i] = n++;
         b[i] = n++;
      }

      #ifdef OUTPUT_MATRIX
         displayMatrix(a, matrixSize, matrixSize);
      #endif

      begin_clock = clock_now();
   }

   // scatters matrix a based on size and number of processors
   MPI_Scatter(a, cellsPerProc, MPI_FLOAT, sentRow, cellsPerProc, MPI_FLOAT, 0, MPI_COMM_WORLD);

   // sends b matrix to all processes
   MPI_Bcast(b, matrixSize * matrixSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

   MPI_Barrier(MPI_COMM_WORLD);

   float *result = calloc(cellsPerProc, sizeof(float));
   if (myRank == 0) begin_clock_mult = clock_now();

   // computes matrix multiplication of sentRow with b in serial
   // sentRow is chunk of matrix A process received
   // result is the resulting chunk matrix from the matrix multiplication
   for (i = 0; i < rowsPerProc; i++) { // loop through the rows each processor is assigned
      for (j = 0; j < matrixSize; j++) { // column offset of c and b
         for (k = 0; k < matrixSize; k++) // column offset of a, row offset of b
            result[j + (i * matrixSize)] += sentRow[i * matrixSize + k] * b[k * matrixSize + j];
      }
   }

   if (myRank == 0) end_clock_mult = clock_now();

   MPI_Barrier(MPI_COMM_WORLD);

   // gathers result chunks from all processes into matrix c
   MPI_Gather(result, cellsPerProc, MPI_FLOAT, c, cellsPerProc, MPI_FLOAT, 0, MPI_COMM_WORLD);

   end_clock = clock_now();
   if (myRank == 0) {
      uint64_t clock_diff = end_clock_mult - begin_clock_mult;
      computation_time = (double) clock_diff / (double) 512000000;
      printf("Parallel multiplication\n");
      #ifdef OUTPUT_MATRIX
         displayMatrix(c, matrixSize, matrixSize);
      #endif
      free(a);
      free(c);

      // time of matrix multiply computation only
      printf("Number of seconds for processes to multiply matrices: %f\n", computation_time);
   }

   free(sentRow);
   free(b);

   remove(FILENAME);

   #ifdef IO
      MPI_Barrier(MPI_COMM_WORLD);

      // before_write_at and after_write_at for measuring time of MPI I/O write operation only,
      // includes both MPI_File_set_atomicity for correctness and MPI_File_write_at
      uint64_t before_write_at, after_write_at;

      // MPI I/O to write chunk to file
      MPI_File fh;
      MPI_Status status;

      if (myRank == 0) {
         printf("\nMPI I/O\n");
      }

      MPI_File_open(MPI_COMM_WORLD, FILENAME,
                  MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);

      if (myRank == 0) {
         before_write_at = clock_now();
      }

      MPI_File_set_atomicity(fh, 1);
      MPI_File_write_at(fh, myRank * cellsPerProc * sizeof(float), result, cellsPerProc, MPI_FLOAT, &status);
      if(myRank == 0) after_write_at = clock_now();
      MPI_Barrier(MPI_COMM_WORLD);

      if (myRank == 0) {
         MPI_File_close(&fh);
         printf("Processes wrote chunks to file.\n");
      } else {
         MPI_File_close(&fh);
      }

      if (myRank == 0) {
         uint64_t write_at_cycles = after_write_at - before_write_at;
         double write_sec = (double) write_at_cycles / (double) 512000000;

         // time of MPI write operation only
         printf("Number of seconds for processes to write chunks to file: %.10f", write_sec);
      }
   #endif

   free(result);
   if(myRank == 0) {
      double total_time = (double) (end_clock - begin_clock) / (double) 512000000;
      double overhead = total_time - computation_time;

      // total execution time of matrix multiply (including overhead time), communication overhead time, and
      // percentage of total execution time that is taken up by overhead
      printf("\nTotal Time: %f seconds, Total communication overhead for multiply: %f seconds, percentage time not multiplying: %f\n", total_time, overhead, overhead * 100 / total_time);
   }
   MPI_Finalize();
   return 0;
}
