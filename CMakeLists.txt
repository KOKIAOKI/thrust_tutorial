cmake_minimum_required(VERSION 3.5)
project(test)

find_package(CUDA REQUIRED)

# helo
CUDA_ADD_EXECUTABLE(hello src/hello.cu)

# thrust sync and nosync test
CUDA_ADD_EXECUTABLE(sync src/sync.cu)

if(CUDA_VERSION VERSION_GREATER "12.0")
    CUDA_ADD_EXECUTABLE(nosync src/nosync.cu)
    CUDA_ADD_EXECUTABLE(hello_async src/hello_async.cu)
else()
    CUDA_ADD_EXECUTABLE(nosync src/oldver_nosync.cu)
    CUDA_ADD_EXECUTABLE(hello_async src/oldver_hello_async.cu)
endif()

# memcpy and memcpyasync test
CUDA_ADD_EXECUTABLE(equal_copy src/additional_test/equal_copy.cu)
CUDA_ADD_EXECUTABLE(equal_copy2 src/additional_test/equal_copy2.cu)
CUDA_ADD_EXECUTABLE(equal_copy3 src/additional_test/equal_copy3.cu)
CUDA_ADD_EXECUTABLE(memcpyasync src/additional_test/memcpyasync.cu)
CUDA_ADD_EXECUTABLE(memcpyasync2 src/additional_test/memcpyasync2.cu)
CUDA_ADD_EXECUTABLE(memcpyasync3 src/additional_test/memcpyasync3.cu)