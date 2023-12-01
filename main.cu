#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <Eigen/Dense>
#include "igl/readOFF.h"
#include "igl/writeOBJ.h"
#include "cuda_runtime.h"

using namespace std;

std::string proj_dir_path;

unsigned num_particles = 0;

//we have to use pos_vis because libigl uses Eigen::MatrixXd
static float3 *pos_d = NULL; // vertex positions in device
static vector<float3> pos_h; // vertex positions in host
Eigen::MatrixXd pos_vis;     // vertex positions for visualization 
Eigen::MatrixXi tri;     

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


__global__ void field_add_one(float3 *pos_d, unsigned num_particles)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles)
    {
        // pos_d[idx].x += 1.0f;
        pos_d[idx].y += 1.0f;
        // pos_d[idx].z += 1.0f;
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

    // copy pos_vis to vector<float3> pos_h
    pos_h.resize(num_particles);
    for (int i = 0; i < num_particles; i++)
    {
        pos_h[i].x = pos_vis(i, 0);
        pos_h[i].y = pos_vis(i, 1);
        pos_h[i].z = pos_vis(i, 2);
    }

    // allocate pos_d and copy pos_h to pos_d
    cudaMalloc((void **)&pos_d, num_particles * sizeof(float3));
    cudaMemcpy(pos_d, pos_h.data(), num_particles * sizeof(float3), cudaMemcpyHostToDevice);
    
    // add one to each vertex
    field_add_one<<<(num_particles + 255) / 256, 256>>>(pos_d, num_particles);

    // copy pos_d to host
    cudaMemcpy(pos_h.data(), pos_d, num_particles * sizeof(float3), cudaMemcpyDeviceToHost);

    // print 10 vertices
    printf("After adding one:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("pos_h[%d] = (%f, %f, %f)\n", i, pos_h[i].x, pos_h[i].y, pos_h[i].z);
    }

    // copy pos_h to pos_vis
    for (int i = 0; i < num_particles; i++)
    {
        pos_vis(i, 0) = pos_h[i].x;
        pos_vis(i, 1) = pos_h[i].y;
        pos_vis(i, 2) = pos_h[i].z;
    }

    igl::writeOBJ(proj_dir_path + "/data/models/bunny2.obj", pos_vis, tri);

    cudaFree(pos_d);
}