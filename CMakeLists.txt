cmake_minimum_required(VERSION 3.5)
project(test)

find_package(CUDA REQUIRED)

CUDA_ADD_EXECUTABLE(hello src/hello.cu)

CUDA_ADD_EXECUTABLE(sync src/sync.cu)

if(CUDA_VERSION VERSION_GREATER "12.0")
    CUDA_ADD_EXECUTABLE(nosync src/nosync.cu)
else()
    CUDA_ADD_EXECUTABLE(nosync src/oldver_nosync.cu)
endif()
