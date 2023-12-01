#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include "igl/readOFF.h"
#include "igl/writeOBJ.h"
#include "cuda_runtime.h"

using namespace std;

std::string proj_dir_path;

static float3 *pos_d = NULL;
static vector<float3> pos_h;
Eigen::MatrixXd pos_vis; // vertex positions
Eigen::MatrixXi tri;     // triangle indices

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


int main(int argc, char *argv[])
{
    get_proj_dir_path();

    // Load a mesh
    igl::readOFF(proj_dir_path + "/data/models/bunny.OFF", pos_vis, tri);

    printf("pos_vis.rows() = %d\n", pos_vis.rows());

    // print 10 vertices
    for (int i = 0; i < 10; i++)
    {
        printf("pos_vis[%d] = (%f, %f, %f)\n", i, pos_vis(i, 0), pos_vis(i, 1), pos_vis(i, 2));
    }

    igl::writeOBJ(proj_dir_path + "/data/models/bunny.obj", pos_vis, tri);
}