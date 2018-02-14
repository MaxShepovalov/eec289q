#include "stdio.h"
#include <iostream>
#include <math.h>

//optimized kernel for small matrix B
__global__ void KroneckerKernelSmall(int M, int N, float* A, float* B, float* C){

  //find position
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  //optimize matrix B (load whole array B to SM)
  extern __shared__ float vs[];
  for (int i = 0; i < N*N; i += blockDim.x) {

    //find index of B to copy from
    int b_ind = i + threadIdx.x;
    if ( b_ind < N*N){
      vs[b_ind] = B[b_ind];
//3 debug lines
//      if (blockIdx.x == 252 and (b_ind == 511 or b_ind == 512)){
//        printf("blk:%d, thr:%d Loads B[%d] check:%f expect:%f\n",blockIdx.x,threadIdx.x,index,vs[b_ind],B[b_ind]);
//      }
    }
  }

  //sync all threads
  __syncthreads();

  //row and column calculation for output from current index
  int row = floorf(index/(M*N));
  int col = index%(M*N);

  //A and B index calc
  int a_idx = floorf(row/N) * M + floorf(col/N);
  int b_idx = (row%N) * N + (col%N);

  //actual compute
  C[index] = A[a_idx] * vs[b_idx];

//4 debug lines
//  if (row==8 and (col==7 or col==8)){
//  if (blockIdx.x == 1 and threadIdx.x == 1){
//    printf("blk:%d, thr:%d, bDIM:%d, gDIM:%d, Performs r:%d c:%d C[%d]=A[%d]*B[%d], C = %f, A=%f, B=%f\n",blockIdx.x,threadIdx.x,blockDim.x,gridDim.x,row,col,index,a_idx,b_idx,C[index],A[a_idx],vs[b_idx]);
//  }
}

//optimized code for small matrix B
void KroneckerGPUSmall(int M, int N, float* A, float* B, float* C){

  //number of threads (512 maximum)
  int nthreads = min(N*M*N*M, 512); //512 maximum

  //number of blocks (grid size), find value to match the given size of data
  int nblocks = ceil(N*M*N*M/nthreads);

  //launch the kernel
  //                     nblocks, nthreads, SM size
  KroneckerKernelSmall<<<nblocks, nthreads, N*N*sizeof(float)>>>(M,N,A,B,C);
}

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
//debug line
//          std::cout << "Processing C[" << rowC << "," << colC << "] with A[" << rowA << "," << colA << "] and B[" << rowB << "," << colB << "]" << std::endl;
          C[rowC * (M * N) + colC] = elemA * elemB;
        }
      }
    }
  }
}

//main
int main(){
  int N,M;
  float *A, *B, *C;

  //set matrices dimentions (<16kB, N < 64) (M+N):
  M = 256; //size of A
  N = 64; //size of B

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
  std::cout << "Small B optimized computation A:" << M  << " B:" << N << " C:" << N*M << std::endl;

  //compute reference
  std::cout << "CPU start" << std::endl;
  KroneckerCPU(M,N,A,B,Ccpu);
  std::cout << "CPU end" << std::endl;

  //compute answer
  std::cout << "GPU start" << std::endl;
  KroneckerGPUSmall(M,N,A,B,C);
  //KroneckerGPU(M,N,A,B,C);

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
  int match = 0;
  for (int row=0; row<M*N; row++){
    for (int columns=0; columns<N*M; columns++){
      int i = row * N * M + columns;
      if (fabs(C[i]-Ccpu[i]) > 0.01) {   //Bus error
        miss++;
        if (miss < 10) {
          std::cout << "row " << row << ", col " << columns;
          std::cout << " Mismatch: GPU " << C[i] << ", CPU " << Ccpu[i] << std::endl;
        } else if (miss == 10) {
          std::cout << "and more ..." << std::endl;
        }
      }
      else {
        match++;
//2 debug lines
//        std::cout << "row " << row << ", col " << columns;
//        std::cout << " OK" << std::endl;
      }
    }
  }
  std::cout << "Found " << miss << " mismatches" << std::endl;
  std::cout << "      " << match << " OK" << std::endl;

//  std::cout << "Search done" << std::endl << "Free pointers" << std::endl;
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  free(Ccpu);
  return 0;
}
