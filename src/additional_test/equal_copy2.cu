#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <chrono>

int main(void)
{
  int multivec_size = 10000;
  int vector_size = 10000;
  std::vector<thrust::host_vector<int>> host_multivec(multivec_size);
  for (int i = 0; i < multivec_size; i++)
  {
    thrust::host_vector<int> a(vector_size);
    thrust::fill(a.begin(), a.end(), i);
    host_multivec[i] = a;
  }

  // Create CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // warming up
  for (int i = 0; i < 10; i++)
  {
    void *ptr;
    cudaMallocAsync(&ptr, sizeof(int) * 1024, stream);
    cudaFreeAsync(ptr, stream);
    cudaMallocHost(&ptr, sizeof(int) * 1024);
    cudaFreeHost(ptr);
  }

  std::vector<thrust::device_vector<int>> device_multivec(multivec_size);
  std::vector<thrust::host_vector<int>> host_output_multivec(multivec_size);

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < multivec_size; i++)
  {
    device_multivec[i].resize(vector_size);
    host_output_multivec[i].resize(vector_size);
  }

  for (int i = 0; i < multivec_size; i++)
  {

    device_multivec[i] = host_multivec[i];
  }

  for (int i = 0; i < multivec_size; i++)
  {
    host_output_multivec[i] = device_multivec[i];
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  double time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  std::cout << "[test]Execution time:" << time << "[msec] " << std::endl;

  /*test*/
  // std::cout << host_output_multivec[multivec_size - 1][vector_size - 1] << std::endl;

  cudaStreamDestroy(stream);
  return 0;
}