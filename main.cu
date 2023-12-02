#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdio>

#include <cuda_runtime.h>
#include <thrust/universal_vector.h>
#include "helper_cuda.h"

#include <Eigen/Dense>
#include "igl/readOFF.h"
#include "igl/writeOBJ.h"


using namespace std;

std::string proj_dir_path;

using Field1f = thrust::universal_vector<float>;
using Field3f = thrust::universal_vector<float3>;
using Field3i = thrust::universal_vector<int3>;

unsigned num_particles = 0;

//we have to use pos_vis because libigl uses Eigen::MatrixXd

Eigen::MatrixXd pos_vis;     // vertex positions for visualization 
Eigen::MatrixXi tri;     

Field3f pos_uni;

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

//learn from https://github.com/parallel101/course/blob/2d30da61b442008c003f69225e6feca20a4ca7df/08/06_thrust/01/main.cu
template <class Func>
__global__ void parallel_for(int n, Func func) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        func(i);
    }
}

int main(int argc, char *argv[])
{
    get_proj_dir_path();

    // Load a mesh
    igl::readOFF(proj_dir_path + "/data/models/bunny.OFF", pos_vis, tri);

    num_particles = pos_vis.rows();

    printf("num_particles = %d\n", num_particles);

    // print 10 vertices
    for (int i = 0; i < 10; i++)
    {
        printf("pos_vis[%d] = (%f, %f, %f)\n", i, pos_vis(i, 0), pos_vis(i, 1), pos_vis(i, 2));
    }

    igl::writeOBJ(proj_dir_path + "/data/models/bunny.obj", pos_vis, tri);

    // copy pos_vis to vector<float3> pos_uni
    pos_uni.resize(num_particles);
    for (int i = 0; i < num_particles; i++)
    {
        pos_uni[i].x = pos_vis(i, 0);
        pos_uni[i].y = pos_vis(i, 1);
        pos_uni[i].z = pos_vis(i, 2);
    }


    // add one to each vertex
    parallel_for<<<num_particles / 512, 128>>>(num_particles, [pos = pos_uni.data()] __device__ (int i) {
        pos[i].y += 1.0;
    });
    checkCudaErrors(cudaDeviceSynchronize());

    // print 10 vertices
    printf("After adding one:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("pos_uni[%d] = (%f, %f, %f)\n", i, pos_uni[i].x, pos_uni[i].y, pos_uni[i].z);
    }

    // copy pos_uni to pos_vis
    for (int i = 0; i < num_particles; i++)
    {
        pos_vis(i, 0) = pos_uni[i].x;
        pos_vis(i, 1) = pos_uni[i].y;
        pos_vis(i, 2) = pos_uni[i].z;
    }

    igl::writeOBJ(proj_dir_path + "/data/models/bunny2.obj", pos_vis, tri);

}