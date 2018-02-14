#include "stdio.h"
#include <iostream>
#include <math.h>

//optimized kernel for small matrix B
__global__ void KroneckerKernelSmall(int M, int N, float* A, float* B, float* C){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  //C row and column calculation
  int row = floorf(index/(M*N));
  int col = index%(M*N);

  //A and B index calc
  int a_idx = floorf(row/N) * M + floorf(col/N);
  int b_idx = (row%N) * N + (col%N);

  //actual compute
  C[index] = A[a_idx] * B[b_idx];

}

//optimized code for small matrix B
void KroneckerGPUSmall(int M, int N, float* A, float* B, float* C){

  std::cout << "Using method for small B" << std::endl;

  //number of threads (512 maximum)
  int nthreads = min(N*M*N*M, 512); //512 maximum

  //number of blocks (grid size), find value to match the given size of data
  int nblocks = floor(N*M*N*M/nthreads);

  //launch the kernel
  KroneckerKernelSmall<<<nblocks, nthreads>>>(M,N,A,B,C);
}

///////////////////////////////////////////////////////////////////////////////////

__global__ void KroneckerKernel(int M, int N, float* A, float* B, float* C){
  float bl1, bl2, bl3, bl4;
  int rowA, colA, rowB, colB, Ci, j;
  //load B to SM
  j = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i=0; i < N*N/4; i++){
    
    //read B and compute C
    bl1 = B[i*4] * A[j];
    bl2 = B[i*4+1] * A[j];
    bl3 = B[i*4+2] * A[j];
    bl4 = B[i*4+3] * A[j];
    
    //write output
    rowA = floorf(j/M);
    rowB = floorf(i*4/N);
    colA = j%M;
    colB = (i*4)%N;
    Ci = N*(rowA*M*N + colA) + rowB*M*N + colB;
    C[Ci] = bl1;
    C[Ci+1] = bl2;
    C[Ci+2] = bl3;
    C[Ci+3] = bl4;
//3 debug lines
//    if (blockIdx.x == 0 and threadIdx.x == 0 and Ci < 20) {
//      printf("blk %d, thr %d, i %d : A %d [%d,%d], B %d [%d,%d], C %d [%d,%d]\n",blockIdx.x,threadIdx.x,i,j,rowA,colA,i*4,rowB,colB,Ci,floorf(Ci/(M*N)),Ci%(M*N));
//    }
  }
}

void KroneckerGPU(int M, int N, float* A, float* B, float* C){

  std::cout << "Using method for large B" << std::endl;

  int nthreads = min(512, M*M);
  int nblocks = ceil(M*M / nthreads);
  //                gridDim  blockDim
  KroneckerKernel<<<nblocks, nthreads>>>(M,N,A,B,C);
}

////////////////////////////////////////////////////////////////////////////////////

//from reference.cpp
void KroneckerCPU(int M, int N, float* A, float* B, float* C){

  for (int rowA = 0; rowA < M; rowA++){

    for (int colA = 0; colA < M; colA++){
      float elemA = A[rowA * M + colA];

      for (int rowB = 0; rowB < N; rowB++){
        int rowC = rowA * N + rowB;

        for (int colB = 0; colB < N; colB++){
          int colC = colA * N + colB;
          float elemB = B[rowB * N + colB];
          C[rowC * (M * N) + colC] = elemA * elemB;
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////
//main
int main(){
  int N,M;
  float *A, *B, *C;

  //set matrices dimentions (<16kB, N <= 64):
  M = 16; //size of A
  N = 128; //size of B

  cudaMallocManaged(&A, M*M*sizeof(float));
  cudaMallocManaged(&B, N*N*sizeof(float));
  cudaError malC = cudaMallocManaged(&C, N*N*M*M*sizeof(float));
  float* Ccpu = (float*) malloc(sizeof(float) * M * N * M * N);

  if (malC != cudaSuccess) {
    std::cout << "Cannot allocate C, err: " << cudaGetErrorName(malC) << std::endl;
    exit(malC);
  }

  //fill arrays
  for (int i=0; i < M*M; i++){
    A[i] = i+1;
  }
  for (int i=0; i < N*N; i++){
    B[i] = i+1;
  }
  for (int i=0; i < N*N*M*M; i++){
    C[i] = 0.0f;
    Ccpu[i] = 0.0f;
  }

  //annonce
  std::cout << "Computing Kronecker with sizes A:" << M  << ", B:" << N << ", result C:" << N*M << std::endl;

  //compute reference
  std::cout << "CPU start" << std::endl;
  KroneckerCPU(M,N,A,B,Ccpu);
  std::cout << "CPU end" << std::endl;

  //compute answer
  std::cout << "GPU start" << std::endl;
  //KroneckerGPUSmall(M,N,A,B,C);
  KroneckerGPU(M,N,A,B,C);

  //sync CUDA and CPU
  cudaError synced = cudaDeviceSynchronize();
  if (synced != cudaSuccess){
    std::cout << "cuda sync ERROR happened " << cudaGetErrorName(synced) << std::endl;
    exit(synced);
  } else {
    std::cout << "cuda sync OK" << std::endl;
  }

  std::cout << "GPU end" << std::endl;

  //print mismatches
  std::cout << "looking for mismatches" << std::endl;
  int miss = 0;
  int pass = 0;
  for (int row=0; row<M*N; row++){
    for (int columns=0; columns<N*M; columns++){
      int i = row * N * M + columns;
      if (fabs(C[i]-Ccpu[i]) > 0.01) {   //Bus error
        miss++;
        if (miss < 10){
          std::cout << "row " << row << ", col " << columns;
          std::cout << " Mismatch: GPU " << C[i] << ", CPU " << Ccpu[i] << std::endl;
        } else if (miss==10){
          std::cout << "and more..." << std::endl;
        }
      } else {
        pass++;
      }
    }
  }
  std::cout << "Found " << miss << " mismatches" << std::endl;
  std::cout << "      " << pass << " OK" << std::endl;

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  free(Ccpu);
  return 0;
}
