#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <filesystem>

#include <cuda_runtime.h>
#include <thrust/universal_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "helper_cuda.h"
#include "helper_math.h"
#include "helper_timer.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

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
using Field1i = thrust::universal_vector<int>;

// global fields
Field3f pos;
Field2i edge;
Field1i tri;
Field1f rest_len;
Field3f vel;
Field1f inv_mass;
Field1f lagrangian;
Field1f constraints;
Field1f dLambda;
Field3f pos_mid;
Field3f acc_pos;
Field3f old_pos;

// we have to use pos_vis for visualization because libigl uses Eigen::MatrixXd
Eigen::MatrixXd pos_vis;
Eigen::MatrixXi tri_vis;

// contorl variables
std::string proj_dir_path;
unsigned num_particles = 0;
unsigned frame_num = 0;
unsigned end_frame = 10;
unsigned max_iter = 50;
std::string out_dir = "./result/cloth3d_256_50_amg/";
bool output_mesh = true;

// utility functions
std::string get_proj_dir_path()
{
    std::filesystem::path p(__FILE__);
    std::filesystem::path prj_path = p.parent_path().parent_path();
    proj_dir_path = prj_path.string();

    std::cout << "Project directory path: " << proj_dir_path << std::endl;
    return proj_dir_path;
}
// this code run before main, in case of user forget to call get_proj_dir_path()
static string proj_dir_path_pre_get = get_proj_dir_path();

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

/// @brief Usage: Timer t("timer_name");
///               t.start();
///               //do something
///               t.end();
class Timer
{
private:
    std::chrono::time_point<std::chrono::steady_clock> m_start;
    std::chrono::time_point<std::chrono::steady_clock> m_end;

public:
    std::string name = "";
    Timer(std::string name = "") : name(name){};
    inline void start()
    {
        m_start = std::chrono::steady_clock::now();
    };
    inline void end(string msg="")
    {
        m_end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = m_end - m_start;
        printf("%s(%s timer): %.0f(ms)\n", msg.c_str(), name.c_str(), elapsed.count());
    };
    inline void reset()
    {
        m_start = std::chrono::steady_clock::now();
        m_end = std::chrono::steady_clock::now();
    };
};
Timer global_timer("global");
Timer t_sim("sim"), t_main("main"), t_substep("substep"), t_init("init");

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

// caution: the tic toc cannot be nested
inline void tic()
{
    global_timer.reset();
    global_timer.start();
}

inline void toc(string message = "")
{
    global_timer.end(message);
    global_timer.reset();
}

void copy_pos_to_pos_vis()
{
    // copy pos to pos_vis
    for (int i = 0; i < num_particles; i++)
    {
        pos_vis(i, 0) = pos[i].x;
        pos_vis(i, 1) = pos[i].y;
        pos_vis(i, 2) = pos[i].z;
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

void semi_euler()
{
    float3 gravity = make_float3(0.0, -0.1, 0.0);
    for (int i = 0; i < num_particles; i++)
    {
        if (inv_mass[i] != 0.0)
        {
            vel[i] += h * gravity;
            old_pos[i] = pos[i];
            pos[i] += h * vel[i];
        }
    }
}

void reset_lagrangian()
{
    for (int i = 0; i < NE; i++)
    {
        lagrangian[i] = 0.0;
    }
}

void reset_accpos()
{
    for (int i = 0; i < num_particles; i++)
    {
        acc_pos[i] = make_float3(0.0, 0.0, 0.0);
    }
}

void solve_constraints_xpbd()
{
    for (int i = 0; i < NE; i++)
    {
        int idx0 = edge[i].x;
        int idx1 = edge[i].y;
        float invM0 = inv_mass[idx0];
        float invM1 = inv_mass[idx1];
        float3 dis = pos[idx0] - pos[idx1];
        float constraint = length(dis) - rest_len[i];
        float3 gradient = normalize(dis);
        float l = -constraint / (invM0 + invM1);
        if (invM0 != 0.0)
        {
            acc_pos[idx0] += invM0 * l * gradient;
        }
        if (invM1 != 0.0)
        {
            acc_pos[idx1] -= invM1 * l * gradient;
        }
    }
}

void update_pos()
{
    for (int i = 0; i < num_particles; i++)
    {
        if (inv_mass[i] != 0.0)
        {
            pos[i] += 0.5 * acc_pos[i];
        }
    }
}

void collision()
{
    for (int i = 0; i < num_particles; i++)
    {
        if (pos[i].z < -2.0)
        {
            pos[i].z = 0.0;
        }
    }
}

void update_vel()
{
    for (int i = 0; i < num_particles; i++)
    {
        if (inv_mass[i] != 0.0)
        {
            vel[i] = (pos[i] - old_pos[i]) / h;
        }
    }
}

void substep_xpbd(int max_iter)
{
    semi_euler();
    reset_lagrangian();
    for (int i = 0; i <= max_iter; i++)
    {
        reset_accpos();
        solve_constraints_xpbd();
        update_pos();
        collision();
    }
    update_vel();
}

void main_loop()
{
    for (frame_num = 0; frame_num <= end_frame; frame_num++)
    {
        printf("---------\n");
        printf("frame_num = %d\n", frame_num);
        
        t_substep.start();
        substep_xpbd(max_iter);
        t_substep.end();

        if (output_mesh)
        {
            tic();
            std::string out_mesh_name = proj_dir_path + "/results/" + std::to_string(frame_num) + ".obj";

            printf("output mesh: %s\n", out_mesh_name.c_str());
            copy_pos_to_pos_vis();
            igl::writeOBJ(out_mesh_name, pos_vis, tri_vis);
            toc("output mesh");
        }

        printf("frame_num = %d done\n", frame_num);
        printf("---------\n\n");
    }
}

void load_R_P()
{
    // load R, P
    Eigen::SparseMatrix<double> R, P;

    Eigen::loadMarket(R, proj_dir_path + "/data/misc/R.mtx");
    Eigen::loadMarket(P, proj_dir_path + "/data/misc/P.mtx");

    std::cout << "R: " << R.rows() << " " << R.cols() << std::endl;
    std::cout << "P: " << P.rows() << " " << P.cols() << std::endl;
}

void resize_fields()
{
    pos.resize(num_particles);
    edge.resize(NE);
    rest_len.resize(NE);
    vel.resize(num_particles);
    inv_mass.resize(num_particles);
    lagrangian.resize(NE);
    constraints.resize(NE);
    dLambda.resize(NE);
    pos_mid.resize(num_particles);
    acc_pos.resize(num_particles);
    old_pos.resize(num_particles);
    tri.resize(3 * NT);

    tri_vis.resize(NT, 3);
    pos_vis.resize(num_particles, 3);
}

void init_pos()
{
    for (int i = 0; i < N + 1; i++)
    {
        for (int j = 0; j < N + 1; j++)
        {
            int idx = i * (N + 1) + j;
            // pos[idx] = ti.Vector([i / N,  j / N, 0.5])  # vertical hang
            pos[idx] = make_float3(i / float(N), 0.5, j / float(N)); // horizontal hang
            inv_mass[idx] = 1.0;
        }
    }
    inv_mass[N] = 0.0;
    inv_mass[NV - 1] = 0.0;
}

void init_tri()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; ++j)
        {
            int tri_idx = 6 * (i * N + j);
            int pos_idx = i * (N + 1) + j;
            if ((i + j) % 2 == 0)
            {
                tri[tri_idx + 0] = pos_idx;
                tri[tri_idx + 1] = pos_idx + N + 2;
                tri[tri_idx + 2] = pos_idx + 1;
                tri[tri_idx + 3] = pos_idx;
                tri[tri_idx + 4] = pos_idx + N + 1;
                tri[tri_idx + 5] = pos_idx + N + 2;
            }
            else
            {
                tri[tri_idx + 0] = pos_idx;
                tri[tri_idx + 1] = pos_idx + N + 1;
                tri[tri_idx + 2] = pos_idx + 1;
                tri[tri_idx + 3] = pos_idx + 1;
                tri[tri_idx + 4] = pos_idx + N + 1;
                tri[tri_idx + 5] = pos_idx + N + 2;
            }
        }
    }

    // reshape tri from 3*NT to (NT, 3)
    for (int i = 0; i < NT; i++)
    {
        int tri_idx = 3 * i;
        int pos_idx = 3 * i;
        tri_vis(i, 0) = tri[tri_idx + 0];
        tri_vis(i, 1) = tri[tri_idx + 1];
        tri_vis(i, 2) = tri[tri_idx + 2];
    }
}

void run_simulation()
{
    printf("run_simulation\n");

    t_sim.start();
    
    t_init.start();
    resize_fields();
    init_pos();
    init_edge();
    init_tri();
    load_R_P();
    t_init.end();

    main_loop();

    t_sim.end();
}

int main(int argc, char *argv[])
{
    t_main.start();

    // Load a mesh
    // igl::readOBJ(proj_dir_path + "/data/models/cloth.obj", pos_vis, tri);
    // num_particles = pos_vis.rows();
    num_particles = NV;
    printf("num_particles = %d\n", num_particles);

    run_simulation();

    // // add one to each vertex
    // parallel_for<<<num_particles / 512, 128>>>(num_particles, [pos = pos.data()] __device__ (int i) {
    //     pos[i].y += 1.0;
    // });
    // checkCudaErrors(cudaDeviceSynchronize());

    copy_pos_to_pos_vis();

    // igl::writeOBJ(proj_dir_path + "/data/models/bunny2.obj", pos_vis, tri);

    t_main.end();
}