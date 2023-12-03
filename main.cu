#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdio>
#include <cmath>

#include <cuda_runtime.h>
#include <thrust/universal_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "helper_cuda.h"
#include "helper_math.h"
#include "helper_timer.h"

#include <Eigen/Dense>
#include "igl/readOBJ.h"
#include "igl/writeOBJ.h"

using namespace std;

// constants
const int N = 256;
const int NV = (N + 1) * (N + 1);
const int NT = 2 * N * N;
const int NE = 2 * N * (N + 1) + N * N;
const float h = 0.01;
const int M = NE;
const int new_M = int(NE / 100);
const float compliance = 1.0e-8;
const float alpha = compliance * (1.0 / h / h);

// typedefs
using Field1f = thrust::universal_vector<float>;
using Field3f = thrust::universal_vector<float3>;
using Field3i = thrust::universal_vector<int3>;
using Field2i = thrust::universal_vector<int2>;

using Field1f_host = thrust::host_vector<float>;
using Field3f_host = thrust::host_vector<float3>;
using Field3i_host = thrust::host_vector<int3>;
using Field2i_host = thrust::host_vector<int2>;

// we have to use pos_vis because libigl uses Eigen::MatrixXd
Eigen::MatrixXd pos_vis; // vertex positions for visualization
Eigen::MatrixXi tri;

// global fields
Field3f pos;
Field2i edge;
Field1f rest_len;

// contorl variables
std::string proj_dir_path;
unsigned num_particles = 0;
unsigned frame_num = 0;
unsigned end_frame = 1000;
unsigned max_iter = 50;
// out_dir = f"./result/cloth3d_256_50_amg/"
std::string out_dir = "./result/cloth3d_256_50_amg/";

// utility functions
std::string get_proj_dir_path()
{
    std::string main_path = __FILE__;

    size_t found = main_path.find_last_of("/\\");
    if (found != std::string::npos)
    {
        proj_dir_path = main_path.substr(0, found);
    }

    std::cout << "Project directory path: " << proj_dir_path << std::endl;
    return proj_dir_path;
}

// learn from https://github.com/parallel101/course/blob/2d30da61b442008c003f69225e6feca20a4ca7df/08/06_thrust/01/main.cu
template <class Func>
__global__ void parallel_for(int n, Func func)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x)
    {
        func(i);
    }
}

/* -------------------------------------------------------------------------- */
/*                            simulation functions                            */
/* -------------------------------------------------------------------------- */
void init_edge()
{
    for (int i = 0; i < N + 1; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int edge_idx = i * N + j;
            int pos_idx = i * (N + 1) + j;
            edge[edge_idx].x = pos_idx;
            edge[edge_idx].y = pos_idx + 1;
        }
    }

    int start = N * (N + 1);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N + 1; j++)
        {
            int edge_idx = start + j * N + i;
            int pos_idx = i * (N + 1) + j;
            edge[edge_idx].x = pos_idx;
            edge[edge_idx].y = pos_idx + N + 1;
        }
    }

    start = 2 * N * (N + 1);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int edge_idx = start + i * N + j;
            int pos_idx = i * (N + 1) + j;
            if ((i + j) % 2 == 0)
            {
                edge[edge_idx].x = pos_idx;
                edge[edge_idx].y = pos_idx + N + 2;
            }
            else
            {
                edge[edge_idx].x = pos_idx + 1;
                edge[edge_idx].y = pos_idx + N + 1;
            }
        }
    }

    for (int i = 0; i < NE; i++)
    {
        int idx1 = edge[i].x;
        int idx2 = edge[i].y;
        float3 p1 = pos[idx1];
        float3 p2 = pos[idx2];
        rest_len[i] = length(p1 - p2);
    }
}

void render_loop()
{

    for (frame_num = 0; frame_num <= end_frame; frame_num++)
    {
    }
}

void run_simulation()
{
    printf("run_simulation\n");

    // create and start timer_sim
    StopWatchInterface *timer_sim = NULL;
    sdkCreateTimer(&timer_sim);
    sdkStartTimer(&timer_sim);// start the timer_sim

    edge.resize(NE);
    rest_len.resize(NE);
    init_edge();

    // render_loop();

    // stop and destroy timer_sim
    sdkStopTimer(&timer_sim);
    printf("%s time: %f (ms)\n",__func__, sdkGetTimerValue(&timer_sim));
    sdkDeleteTimer(&timer_sim);
}

int main(int argc, char *argv[])
{
    // create and start timer_main
    StopWatchInterface *timer_main = NULL;
    sdkCreateTimer(&timer_main);
    sdkStartTimer(&timer_main);// start the timer_main

    get_proj_dir_path();

    // Load a mesh
    igl::readOBJ(proj_dir_path + "/data/models/cloth.obj", pos_vis, tri);

    num_particles = pos_vis.rows();

    printf("num_particles = %d\n", num_particles);

    // copy pos_vis to vector<float3> pos
    pos.resize(num_particles);
    for (int i = 0; i < num_particles; i++)
    {
        pos[i].x = pos_vis(i, 0);
        pos[i].y = pos_vis(i, 1);
        pos[i].z = pos_vis(i, 2);
    }

    run_simulation();

    // // add one to each vertex
    // parallel_for<<<num_particles / 512, 128>>>(num_particles, [pos = pos.data()] __device__ (int i) {
    //     pos[i].y += 1.0;
    // });
    // checkCudaErrors(cudaDeviceSynchronize());

    // copy pos to pos_vis
    for (int i = 0; i < num_particles; i++)
    {
        pos_vis(i, 0) = pos[i].x;
        pos_vis(i, 1) = pos[i].y;
        pos_vis(i, 2) = pos[i].z;
    }

    // print 10 vertices
    printf("print 10 vertices:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("pos[%d] = (%f, %f, %f)\n", i, pos[i].x, pos[i].y, pos[i].z);
    }

    // print 10 lines of edge
    printf("print 10 edges:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("edge[%d] = (%d, %d)\n", i, edge[i].x, edge[i].y);
    }

    // print 10 lines of rest_len
    printf("print 10 rest_len:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("rest_len[%d] = %f\n", i, rest_len[i]);
    }

    // igl::writeOBJ(proj_dir_path + "/data/models/bunny2.obj", pos_vis, tri);


    // stop and destroy timer_main
    sdkStopTimer(&timer_main);
    printf("%s time: %f (ms)\n",__func__, sdkGetTimerValue(&timer_main));
    sdkDeleteTimer(&timer_main);
}