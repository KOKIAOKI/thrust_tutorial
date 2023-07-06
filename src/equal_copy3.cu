#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <chrono>

int main(void)
{
  int loop_size = 10000;
  int vector_size = 10000;
  thrust::host_vector<int> host_input(vector_size);
  thrust::fill(host_input.begin(), host_input.end(), 1);

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

  std::vector<thrust::device_vector<int>> device_multivec(loop_size);
  std::vector<thrust::host_vector<int>> host_multivec(loop_size);
  // host to device
  auto h2dt1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < loop_size; i++)
  {
    device_multivec[i] = host_input;
  }
  auto h2dt2 = std::chrono::high_resolution_clock::now();
  double h2dtime = std::chrono::duration_cast<std::chrono::nanoseconds>(h2dt2 - h2dt1).count() / 1e6;
  std::cout << "[host to device]Execution time:" << h2dtime << "[msec] " << std::endl;

  // device to host
  auto d2ht1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < loop_size; i++)
  {
    host_multivec[i] = device_multivec[i];
  }
  auto d2ht2 = std::chrono::high_resolution_clock::now();
  double d2htime = std::chrono::duration_cast<std::chrono::nanoseconds>(d2ht2 - d2ht1).count() / 1e6;
  std::cout << "[host to device]Execution time:" << d2htime << "[msec] " << std::endl;

  /*test*/
  // std::cout << host_multivec[loop_size - 1][vector_size - 1] << std::endl;

  cudaStreamDestroy(stream);
  return 0;
}