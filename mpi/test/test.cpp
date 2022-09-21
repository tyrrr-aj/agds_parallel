#include <stdlib.h>
#include <stdio.h>
#include <stdio.h>
#include <mpi.h>

#define USE_MPI   
#define SEED 35791246

const long N = 1024 * 10;

int main(int argc, char *argv[])
{
   MPI_Init(&argc,&argv);

   int size, rank;
   MPI_Comm_size(MPI_COMM_WORLD,&size);
   MPI_Comm_rank(MPI_COMM_WORLD,&rank);
   
   int local_len = N / size;

   int *numbers, *sums;
   int* local_numbers = new int[local_len];
   int local_sum;

   if (rank == 0) {
      numbers = new int[N];
      sums = new int[size];

      for (long i = 0; i < N; i++) {
         numbers[i] = i % (local_len);
      }
   }

   double start_time, end_time;

   start_time = MPI_Wtime();

   MPI_Scatter(numbers, local_len, MPI_INT, local_numbers, local_len, MPI_INT, 0, MPI_COMM_WORLD);

   local_sum = 0;
   for (long i = 0; i < N / size; i++) {
      for (long j = 0; j < N / size; j++) {
         local_sum += local_numbers[i];
      }
   }

   MPI_Gather(&local_sum, 1, MPI_INT, sums, 1, MPI_INT, 0, MPI_COMM_WORLD);
   
   end_time = MPI_Wtime();

   if (rank == 0) {
      printf("Running time for %d processes: %.5f\n", size, end_time - start_time);

      delete[] sums;
      delete[] numbers;
   }

   delete[] local_numbers;

   MPI_Finalize();
  
}