# thrust_tutorial
You can test thrust tutorial source code easily.

More information : [Thrust tutorial Web page](https://docs.nvidia.com/cuda/thrust/)

## Version dependent
- CUDA 12.0 or higher  
If Cuda version is less than 12.0, code for older thrust versions is executed.

- gcc 10  
(g++ 10)  

Check it out with the command below.
```
$ gcc --version
```

If you can't build this source code and gcc version is not 10, please install gcc 10.
```
$ sudo apt-get update
$ sudo apt install gcc-10
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10
$ sudo update-alternatives --config gcc
# Choose the selection number that corresponds to gcc10.
$ gcc --version
```
Make sure your gcc version is 10.  
If the build fails even after changing the gcc version, please change g++ version to 10 also.

## Nvidia Driver and Cuda check
- nvidia driver check
```
$ nvidia-smi
```
- cuda check
```
$ nvcc -V
```

## Build
```
$ cd thrust_tutorial/
$ mkdir build && cd build/
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
```

## Test
Move to build directory.
```
$ cd thrust_tutorial/build/
```

### Hello
```
$ ./hello
```

## Nsight Systems test
[Documents (Qiita)](https://qiita.com/koki2022/private/b8df5bc8c0e5669eb0f4)

### Nsight Systems Download
[Nsight Systems](https://developer.nvidia.com/gameworksdownload#?dn=nsight-systems)

```
$ sudo apt install ./Download/(file_name.deb)
```

### sync
```
nsys profile ~/thrust_tutorial/build/sync
```

### Async
```
nsys profile ~/thrust_tutorial/build/nosync
```