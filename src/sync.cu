#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>

int main(void)
{
  int vector_size = 10;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  thrust::device_vector<int> D(vector_size);
  for (int i = 0; i < vector_size; i++)
  {
    thrust::fill(thrust::cuda::par.on(stream), D.begin(), D.end(), 0);
  }

  cudaStreamDestroy(stream);

  return 0;
}