#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//void KroneckerGPU(int M, int N, float* A, float* B, float* C);

void KroneckerCPU(int M, int N, float* A, float* B, float* C) {
  
   for (int rowA = 0; rowA < M; rowA++) {
      for (int colA = 0; colA < M; colA++) {
         float elemA = A[rowA * M + colA];

         for (int rowB = 0; rowB < N; rowB++) {
            int rowC = rowA * N + rowB;

            for (int colB = 0; colB < N; colB++) {
               int colC = colA * N + colB;
               float elemB = B[rowB * N + colB];

               C[rowC * (M * N) + colC] = elemA * elemB;
            }
         }
      }
   }
}

void PrintMatrix(float* matrix, int M, int N) {
   for (int row=0; row<M; row++)
   {
      for(int columns=0; columns<N; columns++)
      {
         printf("%7.3f ", matrix[row * N + columns]);
      }
      printf("\n");
   }
}

int main(int argc, char* argv[]) {
   int M = 32;
   int N = 16;

   float* A = (float* ) malloc(sizeof(float) * M * M);
   float* B = (float* ) malloc(sizeof(float) * N * N);
   float* cpu_result = (float* ) malloc(sizeof(float) * M * N * M * N);
   float* gpu_result = (float* ) malloc(sizeof(float) * M * N * M * N);

   for (int i = 0; i < M * M; i++)
     A[i] = i + 1;

   for (int i = 0; i < N * N; i++)
     B[i] = i + 1;

   //KroneckerGPU(M, N, A, B, gpu_result);
   KroneckerCPU(M, N, A, B, cpu_result);
  
   /*
   for (int i = 0; i < M * N * M * N; i++) {
     if (fabs(gpu_result[i] - cpu_result[i]) > 0.01) {
       printf("Mismatch at index %d: GPU: %f CPU %f\n", i, gpu_result[i], cpu_result[i]);
       
       free(A);
       free(B);
       free(cpu_result);
       free(gpu_result);
       return -1;
     }
   }

   printf("Done %f %f\n", gpu_result[M * N * M * N - 1], cpu_result[M * N * M * N - 1]);
   */ 

   //PrintMatrix(cpu_result, M*N, M*N);

   free(A);
   free(B);
   free(cpu_result);
   free(gpu_result);

   return 0;
}
