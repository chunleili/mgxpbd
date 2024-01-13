
// #if USE_CUDA
#include <cuda_runtime.h>
#include <thrust/universal_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "helper_cuda.h"
#include "helper_math.h"
#include "helper_timer.h"
// #endif


// #if USE_CUDA
/** @brief A parallel for loop. It should be used with a lambda function.
 * Learn from https://github.com/parallel101/course/blob/2d30da61b442008c003f69225e6feca20a4ca7df/08/06_thrust/01/main.cu
 * Usage:
 * // add one to each vertex
 * parallel_for<<<num_particles / 512, 128>>>(num_particles, [pos = pos.data()] __device__ (int i) {
 *    pos[i].y += 1.0;
 * });
 * checkCudaErrors(cudaDeviceSynchronize());
 *
 */
template <typename Func>
__global__ void parallel_for(int n, Func func)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x)
    {
        func(i);
    }
}
// #endif


// #if USE_CUDA
/// @brief Usage: SdkTimer t("timer_name");
///               t.start();
///               //do something
///               t.end();
class SdkTimer
{
private:
    StopWatchInterface *m_timer = NULL;

public:
    std::string name = "";
    SdkTimer(std::string name_ = "") : name(name_)
    {
        sdkCreateTimer(&m_timer);
    }
    SdkTimer::~SdkTimer()
    {
        sdkDeleteTimer(&m_timer);
    }

    inline void start()
    {
        sdkStartTimer(&m_timer);
    }

    inline void end()
    {
        sdkStopTimer(&m_timer);
        printf("%s time elapsed: %.4f(ms)\n", name.c_str(), sdkGetTimerValue(&m_timer));
        sdkResetTimer(&m_timer);
    };

    inline void reset()
    {
        sdkResetTimer(&m_timer);
    };
};
// #endif

__global__ void test_kernel(int *a)
{
    int i = threadIdx.x;
    a[i] = i;
}


void test_cuda()
{
    thrust::host_vector<int> h_vec(100);
    thrust::generate(h_vec.begin(), h_vec.end(), rand);
    thrust::device_vector<int> d_vec = h_vec;
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

    for (int i = 0; i < 100; i++)
    {
        printf("%d ", h_vec[i]);
    }

    int *a;
    cudaMallocManaged(&a, 10 * sizeof(int));
    test_kernel<<<1, 10>>>(a);
    cudaDeviceSynchronize();
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", a[i]);
    }
}