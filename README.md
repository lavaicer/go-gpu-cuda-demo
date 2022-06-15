matrix multiplication by go cuda gpu

> 安装 CUDA/GO/C 环境
>
> 编译 .cu 文件的动态链接库
>
> 在根目录下执行 `go mod tidy`
>
> 执行 go run main.go 1025
>
> 打开 localhost:6006 查看图表

cudaGetDeviceProperties get GPU config return a struct

中文译注(英文见下文)：

```c
struct cudaDeviceProp {
    char name[256]; //器件的名字
    size_t totalGlobalMem; //Global Memory 的 byte 大小
    size_t sharedMemPerBlock; //线程块可以使用的共用记忆体的最大值。byte 为单位，多处理器上的所有线程块可以同时共用这些记忆体
    int regsPerBlock; //线程块可以使用的 32 位寄存器的最大值，多处理器上的所有线程快可以同时实用这些寄存器
    int warpSize; //按线程计算的 wrap 块大小
    size_t memPitch; //做内存复制是可以容许的最大间距，允许通过 cudaMallocPitch（）为包含记忆体区域的记忆提复制函数的最大间距，以 byte 为单位。
    int maxThreadsPerBlock; //每个块中最大线程数
    int maxThreadsDim[3]; //块各维度的最大值
    int maxGridSize[3]; //Grid 各维度的最大值
    size_t totalConstMem; //常量内存的大小
    int major; //计算能力的主代号
    int minor; //计算能力的次要代号
    int clockRate; //时钟频率
    size_t textureAlignment; //纹理的对齐要求
    int deviceOverlap; //器件是否能同时执行 cudaMemcpy()和器件的核心代码
    int multiProcessorCount; //设备上多处理器的数量
    int kernelExecTimeoutEnabled; //是否可以给核心代码的执行时间设置限制
    int integrated; //这个 GPU 是否是集成的
    int canMapHostMemory; //这个 GPU 是否可以讲主 CPU 上的存储映射到 GPU 器件的地址空间
    int computeMode; //计算模式
    int maxTexture1D; //一维 Textures 的最大维度
    int maxTexture2D[2]; //二维 Textures 的最大维度
    int maxTexture3D[3]; //三维 Textures 的最大维度
    int maxTexture2DArray[3]; //二维 Textures 阵列的最大维度
    int concurrentKernels; //GPU 是否支持同时执行多个核心程序
}
```

English Version：

```c
struct device_builtin cudaDeviceProp {
    char name[256]; /**< ASCII string identifying device \*/
    size_t totalGlobalMem; /**< Global memory available on device in bytes _/
    size_t sharedMemPerBlock; /\*\*< Shared memory available per block in bytes _/
    int regsPerBlock; /**< 32-bit registers available per block \*/
    int warpSize; /**< Warp size in threads _/
    size_t memPitch; /\*\*< Maximum pitch in bytes allowed by memory copies _/
    int maxThreadsPerBlock; /**< Maximum number of threads per block \*/
    int maxThreadsDim[3]; /**< Maximum size of each dimension of a block _/
    int maxGridSize[3]; /\*\*< Maximum size of each dimension of a grid _/
    int clockRate; /**< Clock frequency in kilohertz \*/
    size_t totalConstMem; /**< Constant memory available on device in bytes _/
    int major; /\*\*< Major compute capability _/
    int minor; /**< Minor compute capability \*/
    size_t textureAlignment; /**< Alignment requirement for textures _/
    size_t texturePitchAlignment; /\*\*< Pitch alignment requirement for texture references bound to pitched memory _/
    int deviceOverlap; /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. \*/
    int multiProcessorCount; /**< Number of multiprocessors on device _/
    int kernelExecTimeoutEnabled; /\*\*< Specified whether there is a run time limit on kernels _/
    int integrated; /**< Device is integrated as opposed to discrete \*/
    int canMapHostMemory; /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer _/
    int computeMode; /\*\*< Compute mode (See ::cudaComputeMode) _/
    int maxTexture1D; /**< Maximum 1D texture size \*/
    int maxTexture1DMipmap; /**< Maximum 1D mipmapped texture size _/
    int maxTexture1DLinear; /\*\*< Maximum size for 1D textures bound to linear memory _/
    int maxTexture2D[2]; /**< Maximum 2D texture dimensions \*/
    int maxTexture2DMipmap[2]; /**< Maximum 2D mipmapped texture dimensions _/
    int maxTexture2DLinear[3]; /\*\*< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory _/
    int maxTexture2DGather[2]; /**< Maximum 2D texture dimensions if texture gather operations have to be performed \*/
    int maxTexture3D[3]; /**< Maximum 3D texture dimensions _/
    int maxTexture3DAlt[3]; /\*\*< Maximum alternate 3D texture dimensions _/
    int maxTextureCubemap; /**< Maximum Cubemap texture dimensions \*/
    int maxTexture1DLayered[2]; /**< Maximum 1D layered texture dimensions _/
    int maxTexture2DLayered[3]; /\*\*< Maximum 2D layered texture dimensions _/
    int maxTextureCubemapLayered[2];/**< Maximum Cubemap layered texture dimensions \*/
    int maxSurface1D; /**< Maximum 1D surface size _/
    int maxSurface2D[2]; /\*\*< Maximum 2D surface dimensions _/
    int maxSurface3D[3]; /**< Maximum 3D surface dimensions \*/
    int maxSurface1DLayered[2]; /**< Maximum 1D layered surface dimensions _/
    int maxSurface2DLayered[3]; /\*\*< Maximum 2D layered surface dimensions _/
    int maxSurfaceCubemap; /**< Maximum Cubemap surface dimensions \*/
    int maxSurfaceCubemapLayered[2];/**< Maximum Cubemap layered surface dimensions _/
    size_t surfaceAlignment; /\*\*< Alignment requirements for surfaces _/
    int concurrentKernels; /**< Device can possibly execute multiple kernels concurrently \*/
    int ECCEnabled; /**< Device has ECC support enabled _/
    int pciBusID; /\*\*< PCI bus ID of the device _/
    int pciDeviceID; /**< PCI device ID of the device \*/
    int pciDomainID; /**< PCI domain ID of the device _/
    int tccDriver; /\*\*< 1 if device is a Tesla device using TCC driver, 0 otherwise _/
    int asyncEngineCount; /**< Number of asynchronous engines \*/
    int unifiedAddressing; /**< Device shares a unified address space with the host _/
    int memoryClockRate; /\*\*< Peak memory clock frequency in kilohertz _/
    int memoryBusWidth; /**< Global memory bus width in bits \*/
    int l2CacheSize; /**< Size of L2 cache in bytes _/
    int maxThreadsPerMultiProcessor;/\*\*< Maximum resident threads per multiprocessor _/
    int streamPrioritiesSupported; /**< Device supports stream priorities \*/
    int globalL1CacheSupported; /**< Device supports caching globals in L1 _/
    int localL1CacheSupported; /\*\*< Device supports caching locals in L1 _/
    size*t sharedMemPerMultiprocessor; /**< Shared memory available per multiprocessor in bytes \*/
    int regsPerMultiprocessor; /**< 32-bit registers available per multiprocessor */
    int managedMemory; /\*\*< Device supports allocating managed memory on this system \_/
    int isMultiGpuBoard; /**< Device is on a multi-GPU board \*/
    int multiGpuBoardGroupID; /**< Unique identifier for a group of devices on the same multi-GPU board \*/
};
```

---

线程是独立调度和分派的基本单位。线程可以为操作系统内核调度的内核线程，如 Win32 线程；由用户进程自行调度的用户线程，如 Linux 平台的 POSIX Thread；或者由内核与用户进程，如 Windows 7 的线程，进行混合调度。

同一进程中的多条线程将共享该进程中的全部系统资源，如虚拟地址空间，文件描述符和信号处理等等。但同一进程中的多个线程有各自的调用栈（call stack），自己的寄存器环境（register context），自己的线程本地存储（thread-local storage）。

协程非常类似于线程。但是协程是协作式多任务的，而线程典型是抢占式多任务的。这意味着协程提供并发性而非并行性。协程超过线程的好处是它们可以用于硬性实时的语境（在协程之间的切换不需要涉及任何系统调用或任何阻塞调用），这里不需要用来守卫关键区块的同步性原语（primitive）比如互斥锁、信号量等，并且不需要来自操作系统的支持。有可能以一种对调用代码透明的方式，使用抢占式调度的线程实现协程，但是会失去某些利益（特别是对硬性实时操作的适合性和相对廉价的相互之间切换）。

线程是协作式多任务的轻量级线程，本质上描述了同协程一样的概念。其区别，如果一定要说有的话，是协程是语言层级的构造，可看作一种形式的控制流，而线程是系统层级的构造，可看作恰巧没有并行运行的线程。这两个概念谁有优先权是争议性的：线程可看作为协程的一种实现[6]，也可看作实现协程的基底[7]。

Refer:

> https://blog.csdn.net/gggg_ggg/article/details/48130615
>
> > https://hackernoon.com/no-stress-cuda-programming-using-go-and-c-fy1y3agf
>
> > https://zh.wikipedia.org/wiki/%E5%8D%8F%E7%A8%8B
>
> > https://zh.wikipedia.org/wiki/%E7%BA%BF%E7%A8%8B
