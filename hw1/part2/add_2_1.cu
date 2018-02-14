#include <iostream>
#include <math.h>

// function to add the elements of two arrays
__global__
void add(int n, float4 *x, float4 *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride){
      //there are no operators for float4 type
      y[i].x = x[i].x + y[i].x;
      y[i].y = x[i].y + y[i].y;
      y[i].z = x[i].z + y[i].z;
      y[i].w = x[i].w + y[i].w;
  }
}

int main(void)
{
  int N = 1<<20; // 1M elements

  float4 *x, *y;
  cudaMallocManaged(&x, N*sizeof(float4));
  cudaMallocManaged(&y, N*sizeof(float4));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i].x = 1.0f;
    x[i].y = 1.0f;
    x[i].z = 1.0f;
    x[i].w = 1.0f;
    y[i].x = 2.0f;
    y[i].y = 2.0f;
    y[i].z = 2.0f;
    y[i].w = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);

  //sync CUDA and CPU
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++){
    maxError = fmax(maxError, fabs(y[i].x-3.0f));
    maxError = fmax(maxError, fabs(y[i].y-3.0f));
    maxError = fmax(maxError, fabs(y[i].z-3.0f));
    maxError = fmax(maxError, fabs(y[i].w-3.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
