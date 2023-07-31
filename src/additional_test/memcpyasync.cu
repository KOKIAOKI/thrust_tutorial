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
  thrust::device_vector<int> d1(vector_size);
  thrust::device_vector<int> d2(vector_size);
  thrust::device_vector<int> d3(vector_size);
  thrust::device_vector<int> d4(vector_size);
  thrust::device_vector<int> d5(vector_size);
  thrust::device_vector<int> d6(vector_size);
  thrust::device_vector<int> d7(vector_size);
  thrust::device_vector<int> d8(vector_size);
  thrust::device_vector<int> d9(vector_size);
  thrust::device_vector<int> d10(vector_size);

  thrust::host_vector<int> h1(vector_size);
  thrust::host_vector<int> h2(vector_size);
  thrust::host_vector<int> h3(vector_size);
  thrust::host_vector<int> h4(vector_size);
  thrust::host_vector<int> h5(vector_size);
  thrust::host_vector<int> h6(vector_size);
  thrust::host_vector<int> h7(vector_size);
  thrust::host_vector<int> h8(vector_size);
  thrust::host_vector<int> h9(vector_size);
  thrust::host_vector<int> h10(vector_size);

  int size = sizeof(int) * vector_size;
  cudaMemcpyAsync(thrust::raw_pointer_cast(d1.data()), host_input.data(), size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(thrust::raw_pointer_cast(d2.data()), host_input.data(), size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(thrust::raw_pointer_cast(d3.data()), host_input.data(), size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(thrust::raw_pointer_cast(d4.data()), host_input.data(), size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(thrust::raw_pointer_cast(d5.data()), host_input.data(), size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(thrust::raw_pointer_cast(d6.data()), host_input.data(), size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(thrust::raw_pointer_cast(d7.data()), host_input.data(), size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(thrust::raw_pointer_cast(d8.data()), host_input.data(), size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(thrust::raw_pointer_cast(d9.data()), host_input.data(), size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(thrust::raw_pointer_cast(d10.data()), host_input.data(), size, cudaMemcpyHostToDevice, stream);

  cudaMemcpyAsync(h1.data(), thrust::raw_pointer_cast(d1.data()), size, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h2.data(), thrust::raw_pointer_cast(d2.data()), size, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h3.data(), thrust::raw_pointer_cast(d3.data()), size, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h4.data(), thrust::raw_pointer_cast(d4.data()), size, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h5.data(), thrust::raw_pointer_cast(d5.data()), size, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h6.data(), thrust::raw_pointer_cast(d6.data()), size, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h7.data(), thrust::raw_pointer_cast(d7.data()), size, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h8.data(), thrust::raw_pointer_cast(d8.data()), size, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h9.data(), thrust::raw_pointer_cast(d9.data()), size, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h10.data(), thrust::raw_pointer_cast(d10.data()), size, cudaMemcpyDeviceToHost, stream);

  auto t2 = std::chrono::high_resolution_clock::now();
  double time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  std::cout << "[test]Execution time:" << time << "[msec] " << std::endl;

  /*test*/
  // std::cout << h10[vector_size - 1] << std::endl;

  cudaStreamDestroy(stream);

  return 0;
}