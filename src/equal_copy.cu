#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <chrono>

int main(void)
{
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

  auto t1 = std::chrono::high_resolution_clock::now();
  thrust::device_vector<int> d1 = host_input;
  thrust::device_vector<int> d2 = host_input;
  thrust::device_vector<int> d3 = host_input;
  thrust::device_vector<int> d4 = host_input;
  thrust::device_vector<int> d5 = host_input;
  thrust::device_vector<int> d6 = host_input;
  thrust::device_vector<int> d7 = host_input;
  thrust::device_vector<int> d8 = host_input;
  thrust::device_vector<int> d9 = host_input;
  thrust::device_vector<int> d10 = host_input;

  thrust::host_vector<int> h1 = d1;
  thrust::host_vector<int> h2 = d2;
  thrust::host_vector<int> h3 = d3;
  thrust::host_vector<int> h4 = d4;
  thrust::host_vector<int> h5 = d5;
  thrust::host_vector<int> h6 = d6;
  thrust::host_vector<int> h7 = d7;
  thrust::host_vector<int> h8 = d8;
  thrust::host_vector<int> h9 = d9;
  thrust::host_vector<int> h10 = d10;

  auto t2 = std::chrono::high_resolution_clock::now();
  double time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  std::cout << "[test]Execution time:" << time << "[msec] " << std::endl;

  /*test*/
  // std::cout << h10[vector_size - 1] << std::endl;

  cudaStreamDestroy(stream);
  return 0;
}