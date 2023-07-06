#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>

struct Add
{
  __host__ __device__ int operator()(int num)
  {
    return num + 1;
  }
};

int main(void)
{
  int vector_size = 10;
  thrust::host_vector<int> H_in(vector_size);

  // numbering
  for (int i = 0; i < vector_size; i++)
  {
    H_in[i] = i;
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  thrust::device_vector<int> D_in(H_in.size());
  thrust::device_vector<int> D_out(H_in.size());
  thrust::host_vector<int> H_out(H_in.size());
  int size = sizeof(int) * H_in.size();
  cudaMemcpyAsync(thrust::raw_pointer_cast(D_in.data()), H_in.data(), size, cudaMemcpyHostToDevice, stream);

  thrust::transform(thrust::cuda::par_nosync.on(stream), D_in.begin(), D_in.end(), D_out.begin(), Add());

  cudaMemcpyAsync(H_out.data(), thrust::raw_pointer_cast(D_out.data()), size, cudaMemcpyDeviceToHost, stream);

  for (int i = 0; i < H_out.size(); i++)
  {
    std::cout << H_out[i] << ": Hello" << std::endl;
  }

  cudaStreamDestroy(stream);
  return 0;
}