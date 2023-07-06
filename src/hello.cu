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

  thrust::device_vector<int> D_in = H_in;
  thrust::device_vector<int> D_out(D_in.size());

  thrust::transform(D_in.begin(), D_in.end(), D_out.begin(), Add());

  thrust::host_vector<int> H_out = D_out;

  for (int i = 0; i < H_out.size(); i++)
  {
    std::cout << H_out[i] << ": Hello" << std::endl;
  }

  return 0;
}