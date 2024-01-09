#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <vector>
#include <array>
#include <algorithm>
#include <unordered_set>

#include <cuda_runtime.h>
#include <thrust/universal_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "helper_cuda.h"
#include "helper_math.h"
#include "helper_timer.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

using namespace std;
using Eigen::Map;
using Eigen::Vector2i;
using Eigen::Vector3f;
using Eigen::VectorXf;

// constants
const int N = 256;
const int NV = (N + 1) * (N + 1);
const int NT = 2 * N * N;
const int NE = 2 * N * (N + 1) + N * N;
const float h = 0.01;
const int M = NE;
// const int new_M = int(NE / 100);
const float compliance = 1.0e-8;
const float alpha = compliance * (1.0 / h / h);
const float omega = 0.5; // under-relaxing factor

// control variables
std::string proj_dir_path;
unsigned num_particles = 0;
unsigned frame_num = 0;
constexpr unsigned end_frame = 1000;
unsigned max_iter = 50;
std::string out_dir = "./result/cloth3d_256_50_amg/";
bool output_mesh = true;
string solver_type = "GS";
bool should_load_adjacent_edge=true;
float dual_residual[end_frame+1]={0.0};

// typedefs
using Vec3f = Eigen::Vector3f;
using Vec2i = Eigen::Vector2i;
using Vec3i = Eigen::Vector3i;
using Field1f = vector<float>;
using Field3f = vector<Vec3f>;
using Field3i = vector<Vec3i>;
using Field2i = vector<Vec2i>;
using Field1i = vector<int>;
using Field23f = vector<array<Vec3f, 2>>;
using FieldXi = vector<vector<int>>;

// global fields
Field3f pos;
Field2i edge;
Field1i tri;
Field1f rest_len;
Field3f vel;
Field1f inv_mass;
Field1f lagrangian;
Field1f constraints;
Field3f pos_mid;
Field3f acc_pos;
Field3f old_pos;
Field23f gradC;
Field1f b(M);
// Field1f dpos(3*NV);
Field1f dLambda(M);
FieldXi v2e; // vertex to edges
FieldXi adjacent_edge; //give a edge idx, get all its neighbor edges
FieldXi edge_abi; //(a,b,i): vertex a, vertex b, edge i. (a<b)

// we have to use pos_vis for visualization because libigl uses Eigen::MatrixXd
Eigen::MatrixXd pos_vis;
Eigen::MatrixXi tri_vis;

Eigen::SparseMatrix<float> R, P;
Eigen::SparseMatrix<float> M_inv(3 * NV, 3 * NV);
Eigen::SparseMatrix<float> ALPHA(M, M);
Eigen::SparseMatrix<float> A(M, M);
Eigen::SparseMatrix<float> G(M, 3 * NV);
// Eigen::VectorXf dpos(3*NV);

// utility functions
#if defined(WIN32) || defined(_WIN32) || defined(WIN64)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE __attribute__((always_inline))
#endif

FORCE_INLINE float length(const Vec3f &vec)
{
    // return glm::length(vec);
    return vec.norm();
}

FORCE_INLINE Vec3f normalize(const Vec3f &vec)
{
    // return glm::normalize(vec);
    return vec.normalized();
}

FORCE_INLINE float dot(const Vec3f &vec1, const Vec3f &vec2)
{
    // return glm::dot(vec1, vec2);
    return vec1.dot(vec2);
}

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
    inline void end(string msg = "", string unit = "ms")
    {
        m_end = std::chrono::steady_clock::now();
        if (unit == "ms")
        {
            std::chrono::duration<double, std::milli> elapsed = m_end - m_start;
            printf("%s(%s): %.5f(ms)\n", msg.c_str(), name.c_str(), elapsed.count());
        }
        else if (unit == "s")
        {
            std::chrono::duration<double> elapsed = m_end - m_start;
            printf("%s(%s): %.0f(s)\n", msg.c_str(), name.c_str(), elapsed.count());
        }
    }
    inline void reset()
    {
        m_start = std::chrono::steady_clock::now();
        m_end = std::chrono::steady_clock::now();
    };
};
Timer global_timer("global");
Timer t_sim("sim"), t_main("main"), t_substep("substep"), t_init("init"), t_iter("iter");

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
        pos_vis(i, 0) = pos[i][0];
        pos_vis(i, 1) = pos[i][1];
        pos_vis(i, 2) = pos[i][2];
    }
}

void savetxt(string filename, FieldXi &field)
{
    ofstream myfile;
    myfile.open(filename);
    for(auto &i:field)
    {
        for(auto &ii:i)
        {
            myfile << ii << " ";
        }
        myfile << endl;
    }
    myfile.close();
}

void savetxt(string filename, Field2i &field)
{
    ofstream myfile;
    myfile.open(filename);
    for(auto &i:field)
    {
        for(auto &ii:i)
        {
            myfile << ii << " ";
        }
        myfile << endl;
    }
    myfile.close();
}

void savetxt(string filename, Field1f &field)
{
    Eigen::Map<Eigen::VectorXf> v(field.data(), field.size());
    Eigen::saveMarket(v, filename);
}



void loadtxt(std::string filename, FieldXi &M)
{
  printf("Loading %s with FieldXi\n", filename.c_str());
  std::ifstream inputFile(filename);
  std::string line;
//   std::vector<vec> values;
  unsigned int rows = 0;
  while (std::getline(inputFile, line))
  {
    std::istringstream iss(line);
    int val;
    M.resize(rows + 1);
    while (iss >> val)
    {
      M[rows].push_back(val);
    }
    rows++;
  }
//   M.resize(rows);
//   for (int i = 0; i < rows; i++)
//   {
//     unsigned int num_per_row = M[i].size();
//     for (int j = 0; j < num_per_row; j++)
//     {
//       M[i][j] = values[i * num_per_row + j];
//     }
//   }
}


void test()
{
    // Eigen::saveMarket(M_inv, "M.mtx");
    // Eigen::saveMarket(ALPHA, "ALPHA.mtx");
    // Eigen::saveMarket(R, "RR.mtx");
    // Eigen::saveMarket(P, "PP.mtx");
    printf("\nsaving A.mtx\n");
    Eigen::saveMarket(A, "A.mtx");
    // Eigen::saveMarket(G, "G.mtx");
}

float maxField(std::vector<Vec3f> &field)
{
    auto max = field[0][0];
    for (unsigned int i = 1; i < field.size(); i++)
    {
        for (unsigned int j = 0; j < 3; j++)
        {
            if (field[i][j] > max)
                max = field[i][j];
        }
    }
    return max;
}

/**
 * @brief 保存向量场到txt
 *
 * @tparam T
 * @param fileName 文件名
 * @param content 要打印的场
 * @param precision 精度（默认小数点后8位数）
 */
template <typename T>
void printVectorField(std::string fileName, T content, size_t precision = 8)
{
    std::ofstream f;
    f.open(fileName);
    for (const auto &x : content)
    {
        for (const auto &xx : x)
            f << std::fixed << std::setprecision(precision) << xx << "\t";
        f << "\n";
    }
    f.close();
}

/**
 * @brief 保存标场到txt
 *
 * @tparam T
 * @param fileName 文件名
 * @param content 要打印的场
 * @param precision 精度（默认小数点后8位数）
 */
template <typename T>
void printScalarField(std::string fileName, T content, size_t precision = 8)
{
    std::ofstream f;
    f.open(fileName);
    for (const auto &x : content)
    {
        f << std::fixed << std::setprecision(precision) << x << "\n";
    }
    f.close();
}

void write_obj(std::string name = "")
{
    tic();
    std::string path = proj_dir_path + "/results/";
    std::string out_mesh_name = path + std::to_string(frame_num) + ".obj";
    if (name != "")
    {
        out_mesh_name = path + name + std::to_string(frame_num) + ".obj";
    }

    printf("output mesh: %s\n", out_mesh_name.c_str());
    copy_pos_to_pos_vis();
    igl::writeOBJ(out_mesh_name, pos_vis, tri_vis);
    toc("output mesh");
}


void remove_duplicate(std::vector<int> &vec)
{
    std::unordered_set<int> s;
    for (int i : vec)
        s.insert(i);
    vec.assign( s.begin(), s.end() );
    sort( vec.begin(), vec.end() );
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
            edge[edge_idx][0] = pos_idx;
            edge[edge_idx][1] = pos_idx + 1;
        }
    }

    int start = N * (N + 1);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N + 1; j++)
        {
            int edge_idx = start + j * N + i;
            int pos_idx = i * (N + 1) + j;
            edge[edge_idx][0] = pos_idx;
            edge[edge_idx][1] = pos_idx + N + 1;
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
                edge[edge_idx][0] = pos_idx;
                edge[edge_idx][1] = pos_idx + N + 2;
            }
            else
            {
                edge[edge_idx][0] = pos_idx + 1;
                edge[edge_idx][1] = pos_idx + N + 1;
            }
        }
    }

    for (int i = 0; i < NE; i++)
    {
        int idx1 = edge[i][0];
        int idx2 = edge[i][1];
        Vec3f p1 = pos[idx1];
        Vec3f p2 = pos[idx2];
        rest_len[i] = length(p1 - p2);
    }
}

void semi_euler()
{
    Vec3f gravity = Vec3f(0.0, -0.1, 0.0);
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


/// @brief  input a vertex index, output the edge index, maintaining v2e field
void init_v2e()
{
    v2e.resize(num_particles);

    for(int i=0; i<edge.size(); i++)
    {
        int idx0 = edge[i][0];
        int idx1 = edge[i][1];
        
        v2e[idx0].push_back(i);
        v2e[idx1].push_back(i);

        remove_duplicate(v2e[idx0]);
        remove_duplicate(v2e[idx1]);

    }
}

// (a,b,i): vertex a, vertex b, edge i (a<b)
void init_edge_abi()
{
    edge_abi.resize(NE);
    for(int i=0; i<NE; i++)
    {
        edge_abi[i].resize(3);
        edge_abi[i][0] = std::min(edge[i][0],edge[i][1]);
        edge_abi[i][1] = std::max(edge[i][0],edge[i][1]);
        edge_abi[i][2] = i;
    }

    std::sort(edge_abi.begin(), edge_abi.end(), 
    [](const vector<int>& a, const vector<int>& b)
    {
        return a[0]<b[0];
    });
}

void init_adjacent_edge()
{
    if (should_load_adjacent_edge)
    {
        printf("load adjacent edge!\n");
        loadtxt(proj_dir_path+"/data/misc/adjacent_edge.txt",adjacent_edge);
        return;
    }



    tic();
    adjacent_edge.resize(NE);

    int maxsize=0;
    for(int i=0; i<NE; i++)
    {
        int a=edge[i][0];
        int b=edge[i][1];
        for(int j=i+1; j < NE; j++)
        {
            if(j==i)
                continue;
        
            int a1=edge[j][0];
            int b1=edge[j][1];
            if(a==a1||a==b1||b==a1||b==b1)
            {
                adjacent_edge[i].push_back(j);
                adjacent_edge[j].push_back(i);
            }
            if(adjacent_edge[i].size()>maxsize)
            {
                maxsize=adjacent_edge[i].size();
            }
        }
    }
    toc("adjacent");
    printf("maxsize = %d\n", maxsize);


    // //get adjacent edges with shared vertex a
    // std::sort(edge_abi.begin(), edge_abi.end(), 
    // [](const vector<int>& a, const vector<int>& b)
    // {
    //     return a[0]<b[0];
    // });

    // for (int k = 0; k < NE-1; k++)
    // {
    //     int a = edge_abi[k][0];
    //     int b = edge_abi[k][1];
    //     int i = edge_abi[k][2];

    //     int a2 = edge_abi[k+1][0];
    //     int b2 = edge_abi[k+1][1];
    //     int i2 = edge_abi[k+1][2];

    //     if (a==a2)
    //     {
    //         adjacent_edge[i].push_back(i2);
    //         adjacent_edge[i2].push_back(i);
    //     }
    //     k++;
    // }
    
    // //get adjacent edges with shared vertex b
    // std::sort(edge_abi.begin(), edge_abi.end(), 
    // [](const vector<int>& a, const vector<int>& b)
    // {
    //     return a[1]<b[1];
    // });

    // for (int k = 0; k < NE-1; k++)
    // {
    //     int a = edge_abi[k][0];
    //     int b = edge_abi[k][1];
    //     int i = edge_abi[k][2];

    //     int a2 = edge_abi[k+1][0];
    //     int b2 = edge_abi[k+1][1];
    //     int i2 = edge_abi[k+1][2];

    //     if (b==b2)
    //     {
    //         adjacent_edge[i].push_back(i2);
    //         adjacent_edge[i2].push_back(i);
    //     }
    // }
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
        acc_pos[i] = Vec3f(0.0, 0.0, 0.0);
    }
}

void solve_constraints_xpbd()
{
    for (int i = 0; i < NE; i++)
    {
        int idx0 = edge[i][0];
        int idx1 = edge[i][1];
        float invM0 = inv_mass[idx0];
        float invM1 = inv_mass[idx1];
        Vec3f dis = pos[idx0] - pos[idx1];
        float constraint = length(dis) - rest_len[i];
        Vec3f gradient = normalize(dis);
        float l = -constraint / (invM0 + invM1);
        float delta_lagrangian = -(constraint + lagrangian[i] * alpha) / (invM0 + invM1 + alpha);
        lagrangian[i] += delta_lagrangian;
        if (invM0 != 0.0)
        {
            acc_pos[idx0] += invM0 * delta_lagrangian * gradient;
        }
        if (invM1 != 0.0)
        {
            acc_pos[idx1] -= invM1 * delta_lagrangian * gradient;
        }
    }
}

void update_pos()
{
    for (int i = 0; i < num_particles; i++)
    {
        if (inv_mass[i] != 0.0)
        {
            pos[i] += omega * acc_pos[i];
        }
    }
}

void collision()
{
    for (int i = 0; i < num_particles; i++)
    {
        if (pos[i][2] < -2.0)
        {
            pos[i][2] = 0.0;
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

void substep_xpbd()
{
    semi_euler();
    reset_lagrangian();
    for (int i = 0; i <= max_iter; i++)
    {
        // printf("iter = %d\n", i);
        reset_accpos();
        solve_constraints_xpbd();
        update_pos();
        collision();
    }
    update_vel();
}

void fill_M_inv()
{
    typedef Eigen::Triplet<float> T;

    std::vector<T> inv_mass_3(3 * NV);
    for (int i = 0; i < 3 * NV; i++)
    {
        inv_mass_3[i] = T(i, i, inv_mass[int(i / 3)]);
    }
    M_inv.setFromTriplets(inv_mass_3.begin(), inv_mass_3.end());
    M_inv.makeCompressed();
}

void fill_ALPHA()
{
    typedef Eigen::Triplet<float> T;

    std::vector<T> alpha_(NE);
    for (int i = 0; i < NE; i++)
    {
        alpha_[i] = T(i, i, alpha);
    }
    ALPHA.setFromTriplets(alpha_.begin(), alpha_.end());
    ALPHA.makeCompressed();
}

void compute_C_and_gradC()
{
    for (int i = 0; i < NE; i++)
    {
        int idx0 = edge[i][0];
        int idx1 = edge[i][1];
        Vec3f dis = pos[idx0] - pos[idx1];
        constraints[i] = length(dis) - rest_len[i];
        Vec3f g = normalize(dis);

        gradC[i][0] = g;
        gradC[i][1] = -g;
    }
}

void fill_gradC_triplets()
{
    typedef Eigen::Triplet<float> T;

    std::vector<T> gradC_triplets;
    gradC_triplets.reserve(6 * NE);
    int cnt = 0;
    for (int j = 0; j < NE; j++)
    {
        auto ind = edge[j];
        for (int p = 0; p < 2; p++)
        {
            for (int d = 0; d < 3; d++)
            {
                int pid = ind[p];
                gradC_triplets.push_back(T(j, 3 * pid + d, gradC[j][p][d]));
                cnt++;
            }
        }
    }
    // printf("cnt: %d", cnt);
    G.setFromTriplets(gradC_triplets.begin(), gradC_triplets.end());
    G.makeCompressed();
}

void fill_b()
{
    for (int i = 0; i < NE; i++)
    {
        b[i] = -constraints[i] - alpha * lagrangian[i];
    }
}

void calc_dual_residual(int iter)
{
    dual_residual[iter] = 0.0;
    for (int i = 0; i < NE; i++)
    {
        float r = -constraints[i] - alpha * lagrangian[i];
        dual_residual[iter] += r*r;
    }
    dual_residual[iter] = std::sqrt(dual_residual[iter]);
}

// the bug of fill_A_cuda is not fix yet
// void fill_A_cuda()
// {
//     typedef Eigen::Triplet<float> T;

//     std::vector<T> val;
//     val.reserve(12*NE);

//     // add one to each vertex
//     parallel_for<<<NE / 128, 128>>>(num_particles, 
//     [val = val.data()] __device__ (int i) 
//     {
//         //fill diagonal:m1 + m2 + alpha
//         int ii0 = edge[i][0];
//         int ii1 = edge[i][1];
//         float invM0 = inv_mass[ii0];
//         float invM1 = inv_mass[ii1];
//         float diag = (invM0 + invM1 + alpha);
//         val.push_back(T(i, i, diag));

//         //fill off-diagonal: m_a*dot(g_ab,g_ab)
//         vector<int> adj = adjacent_edge[i];
//         for (int j = 0; j < adj.size(); j++)
//         {
//             int adj_edge_idx = adj[j];
//             if(adj_edge_idx==i)
//             {
//                 printf("%d self!\n",adj_edge_idx);
//                 continue;
//             }

//             int jj0 = edge[adj_edge_idx][0];
//             int jj1 = edge[adj_edge_idx][1];

//             // a is shared vertex 
//             // a-b is the first edge, a-c is the second edge
//             int a=-1,b=-1,c=-1;
//             if(ii0==jj0)
//             {
//                 a=ii0;
//                 b=ii1;
//                 c=jj1;
//             }
//             else if(ii0==jj1)
//             {
//                 a=ii0;
//                 b=ii1;
//                 c=jj0;
//             }
//             else if(ii1==jj0)
//             {
//                 a=ii1;
//                 b=ii0;
//                 c=jj1;
//             }
//             else if(ii1==jj1)
//             {
//                 a=ii1;
//                 b=ii0;
//                 c=jj0;
//             }
//             else
//             {
//                 printf("%d no shared vertex!\n",adj_edge_idx);
//                 continue;
//             }
            
            
//             // m_a*dot(g_ab,g_ab)
//             Vec3f g_ab = normalize(pos[a] - pos[b]);
//             Vec3f g_ac = normalize(pos[a] - pos[c]);
//             float off_diag = inv_mass[a] * dot(g_ab, g_ac);

//             val.push_back(T(i, adj_edge_idx, off_diag));
//         }
//     });
//     checkCudaErrors(cudaDeviceSynchronize());
  
//     A.setFromTriplets(val.begin(), val.end());
//     A.makeCompressed();
// }


void fill_A_by_insert()
{
    // typedef Eigen::Triplet<float> T;
    // std::vector<T> val;
    // val.reserve(15*NE);
    A.reserve(Eigen::VectorXf::Constant(M, 15));
    tic();

    for (int i = 0; i < NE; i++)
    {   
        //fill diagonal:m1 + m2 + alpha
        int ii0 = edge[i][0];
        int ii1 = edge[i][1];
        float invM0 = inv_mass[ii0];
        float invM1 = inv_mass[ii1];
        float diag = (invM0 + invM1 + alpha);
        // val.push_back(T(i, i, diag));
        A.insert(i,i) = diag;
        // A.coeffRef(i,i) = diag;

        //fill off-diagonal: m_a*dot(g_ab,g_ab)
        vector<int> adj = adjacent_edge[i];
        for (int j = 0; j < adj.size(); j++)
        {
            int adj_edge_idx = adj[j];
            if(adj_edge_idx==i)
            {
                printf("%d self!\n",adj_edge_idx);
                continue;
            }

            int jj0 = edge[adj_edge_idx][0];
            int jj1 = edge[adj_edge_idx][1];

            // a is shared vertex 
            // a-b is the first edge, a-c is the second edge
            int a=-1,b=-1,c=-1;
            if(ii0==jj0)
            {
                a=ii0;
                b=ii1;
                c=jj1;
            }
            else if(ii0==jj1)
            {
                a=ii0;
                b=ii1;
                c=jj0;
            }
            else if(ii1==jj0)
            {
                a=ii1;
                b=ii0;
                c=jj1;
            }
            else if(ii1==jj1)
            {
                a=ii1;
                b=ii0;
                c=jj0;
            }
            else
            {
                printf("%d no shared vertex!\n",adj_edge_idx);
                continue;
            }
            
            
            // m_a*dot(g_ab,g_ab)
            Vec3f g_ab = normalize(pos[a] - pos[b]);
            Vec3f g_ac = normalize(pos[a] - pos[c]);
            float off_diag = inv_mass[a] * dot(g_ab, g_ac);

            // val.push_back(T(i, adj_edge_idx, off_diag));
            A.insert(i,adj_edge_idx) = off_diag;
            // A.coeffRef(i,adj_edge_idx) = off_diag;
        }

    }
    // A.setFromTriplets(val.begin(), val.end());
    A.makeCompressed();

    // toc("fill A");
    // exit(0);
}


void fill_A()
{
    typedef Eigen::Triplet<float> T;

    std::vector<T> val;
    val.reserve(15*NE);
    for (int i = 0; i < NE; i++)
    {
        //fill diagonal:m1 + m2 + alpha
        int ii0 = edge[i][0];
        int ii1 = edge[i][1];
        float invM0 = inv_mass[ii0];
        float invM1 = inv_mass[ii1];
        float diag = (invM0 + invM1 + alpha);
        val.push_back(T(i, i, diag));

        //fill off-diagonal: m_a*dot(g_ab,g_ab)
        vector<int> adj = adjacent_edge[i];
        for (int j = 0; j < adj.size(); j++)
        {
            int adj_edge_idx = adj[j];
            if(adj_edge_idx==i)
            {
                printf("%d self!\n",adj_edge_idx);
                continue;
            }

            int jj0 = edge[adj_edge_idx][0];
            int jj1 = edge[adj_edge_idx][1];

            // a is shared vertex 
            // a-b is the first edge, a-c is the second edge
            int a=-1,b=-1,c=-1;
            if(ii0==jj0)
            {
                a=ii0;
                b=ii1;
                c=jj1;
            }
            else if(ii0==jj1)
            {
                a=ii0;
                b=ii1;
                c=jj0;
            }
            else if(ii1==jj0)
            {
                a=ii1;
                b=ii0;
                c=jj1;
            }
            else if(ii1==jj1)
            {
                a=ii1;
                b=ii0;
                c=jj0;
            }
            else
            {
                printf("%d no shared vertex!\n",adj_edge_idx);
                continue;
            }
            
            
            // m_a*dot(g_ab,g_ab)
            Vec3f g_ab = normalize(pos[a] - pos[b]);
            Vec3f g_ac = normalize(pos[a] - pos[c]);
            float off_diag = inv_mass[a] * dot(g_ab, g_ac);

            val.push_back(T(i, adj_edge_idx, off_diag));
        }

    }
    A.setFromTriplets(val.begin(), val.end());
    A.makeCompressed();
}


/*
 * Perform one iteration of Gauss-Seidel relaxation on the linear
 * system Ax = b, where A is stored in CSR format and x and b
 * are column vectors.
 *
 * Parameters
 * ----------
 * Ap : array
 *     CSR row pointer
 * Aj : array
 *     CSR index array
 * Ax : array
 *     CSR data array
 * x : array, inplace
 *     approximate solution
 * b : array
 *     right hand side
 * row_start : int
 *     beginning of the sweep
 * row_stop : int
 *     end of the sweep (i.e. one past the last unknown)
 * row_step : int
 *     stride used during the sweep (may be negative)
 *
 * Returns
 * -------
 * Nothing, x will be modified inplace
 *
 * Notes
 * -----
 * The unknowns are swept through according to the slice defined
 * by row_start, row_end, and row_step.  These options are used
 * to implement standard forward and backward sweeps, or sweeping
 * only a subset of the unknowns.  A forward sweep is implemented
 * with gauss_seidel(Ap, Aj, Ax, x, b, 0, N, 1) where N is the
 * number of rows in matrix A.  Similarly, a backward sweep is
 * implemented with gauss_seidel(Ap, Aj, Ax, x, b, N, -1, -1).
// from https://github.com/pyamg/pyamg/blob/0431f825d7e6683c208cad20572e92fc0ef230c1/pyamg/amg_core/relaxation.h#L45
// I=int, T=float, F=float
*/
template <class I = int, class T = float, class F = float>
void gauss_seidel(const I Ap[], const int Ap_size,
                  const I Aj[], const int Aj_size,
                  const T Ax[], const int Ax_size,
                  T x[], const int x_size,
                  const T b[], const int b_size,
                  const I row_start,
                  const I row_stop,
                  const I row_step)
{
    for (I i = row_start; i != row_stop; i += row_step)
    {
        I start = Ap[i];
        I end = Ap[i + 1];
        T rsum = 0.0;
        T diag = 0.0;

        for (I jj = start; jj < end; jj++)
        {
            I j = Aj[jj];
            if (i == j)
                diag = Ax[jj];
            else
                rsum += Ax[jj] * x[j];
        }

        if (diag != (F)0.0)
        {
            x[i] = (b[i] - rsum) / diag;
        }
    }
}

void incre_lagrangian()
{
    for (int i = 0; i < NE; i++)
    {
        lagrangian[i] += dLambda[i];
    }
}

// void add_dpos(const Eigen::VectorXf& dpos)
// {
//     auto dpos3 = dpos.reshaped(NV,3);
//     Vec3f dpos3i=Vec3f(0.0, 0.0, 0.0);
//     for(int i=0; i < num_particles; i++)
//     {
//         dpos3i = dpos3.row(i);
//         pos[i] += dpos3i;
//     }
// }

// void copy_dlambda(Eigen::VectorXf &dLambda_eigen, const Field1f &dLambda)
// {
//     for(int i=0; i < NE; i++)
//     {
//         dLambda_eigen[i] = dLambda[i];
//     }
// }

// void transfer_back_to_pos_matrix()
// {
//     // transfer back to pos
//     for (int i = 0; i < NE; i++)
//     {
//         lagrangian[i] += dLambda[i];
//     }

//     Eigen::Map<Eigen::VectorXf> dLambda_eigen(dLambda.data(), dLambda.size());

//     Eigen::VectorXf dpos_ = M_inv * G.transpose() * dLambda_eigen;

//     // add dpos to pos
//     for (int i = 0; i < num_particles; i++)
//     {
//         pos[i] = pos_mid[i] + Vec3f(dpos_[3 * i], dpos_[3 * i + 1], dpos_[3 * i + 2]);
//     }
// }

void transfer_back_to_pos_mfree()
{
    reset_accpos();

    for (int i = 0; i < NE; i++)
    {
        int idx0 = edge[i][0];
        int idx1 = edge[i][1];
        float invM0 = inv_mass[idx0];
        float invM1 = inv_mass[idx1];
        float delta_lagrangian = dLambda[i];
        Vec3f dis = pos[idx0] - pos[idx1];
        Vec3f gradient = normalize(dis);
        lagrangian[i] += delta_lagrangian;
        if (invM0 != 0.0)
        {
            acc_pos[idx0] += invM0 * delta_lagrangian * gradient;
        }
        if (invM1 != 0.0)
        {
            acc_pos[idx1] -= invM1 * delta_lagrangian * gradient;
        }
    }

    update_pos();
}

void fill_A_add_alpha()
{
    for(int i=0; i<M; i++)
    {
        A.coeffRef(i,i) += alpha;
    }
}

void update_constraints()
{
    for (int i = 0; i < NE; i++)
    {
        int idx0 = edge[i][0];
        int idx1 = edge[i][1];
        Vec3f dis = pos[idx0] - pos[idx1];
        constraints[i] = length(dis) - rest_len[i];
    }
}

void substep_all_solver()
{
    printf("\n\n----frame_num:%d----\n", frame_num);
    semi_euler();
    reset_lagrangian();
    for (int iter = 0; iter <= max_iter; iter++)
    {
        t_iter.start();
        printf("iter = %d", iter);

        
        // assemble A and b
        // compute_C_and_gradC();
        // fill_gradC_triplets();
        // G.makeCompressed();
        // A =  G * M_inv * G.transpose();
        // fill_A_add_alpha();

        fill_A();

        update_constraints();
        fill_b();   //-C-alpha*lagrangian

        // solve Ax=b
        if (solver_type == "GS")
        {
            int max_GS_iter = 1;
            std::fill(dLambda.begin(), dLambda.end(), 0.0);
            for (int GS_iter = 0; GS_iter < max_GS_iter; GS_iter++)
            {
                gauss_seidel<int, float, float>(A.outerIndexPtr(), A.outerSize(),
                                                A.innerIndexPtr(), A.innerSize(), A.valuePtr(), A.nonZeros(),
                                                dLambda.data(), dLambda.size(), b.data(), b.size(), 0, M, 1);
            }
        }

        transfer_back_to_pos_mfree();

        calc_dual_residual(iter);
        printf(" dual_residual = %f\n", dual_residual[iter]);

        t_iter.end();
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
        // substep_xpbd();
        substep_all_solver();
        t_substep.end();

        if (output_mesh)
        {
            write_obj();
        }

        printf("frame_num = %d done\n", frame_num);
        printf("---------\n\n");
    }
}

void load_R_P()
{
    // load R, P
    Eigen::loadMarket(R, proj_dir_path + "/data/misc/R.mtx");
    Eigen::loadMarket(P, proj_dir_path + "/data/misc/P.mtx");

    std::cout << "R: " << R.rows() << " " << R.cols() << std::endl;
    std::cout << "P: " << P.rows() << " " << P.cols() << std::endl;
}

void resize_fields()
{
    pos.resize(num_particles, Vector3f::Zero());
    edge.resize(NE, Vector2i::Zero());
    rest_len.resize(NE, 0.0);
    vel.resize(num_particles, Vector3f::Zero());
    inv_mass.resize(num_particles, 0.0);
    lagrangian.resize(NE, 0.0);
    constraints.resize(NE, 0.0);
    pos_mid.resize(num_particles, Vector3f::Zero());
    acc_pos.resize(num_particles, Vector3f::Zero());
    old_pos.resize(num_particles, Vector3f::Zero());
    tri.resize(3 * NT, 0);
    gradC.resize(NE, array<Vec3f, 2>{Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 0.0)});

    // dpos.resize(3 * NV, 0.0);
    dLambda.resize(M, 0.0);
    b.resize(M, 0.0);

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
            pos[idx] = Vec3f(i / float(N), 0.5, j / float(N)); // horizontal hang
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

void initialization()
{
    t_init.start();
    resize_fields();
    init_pos();
    init_edge();
    init_tri();
    load_R_P();
    fill_M_inv();
    fill_ALPHA();
    init_v2e();
    init_edge_abi();
    init_adjacent_edge();
    // savetxt("adjacent_edge.txt", adjacent_edge);
    // exit(0);
    
    t_init.end();
}

void run_simulation()
{
    printf("run_simulation\n");

    t_sim.start();
    initialization();
    main_loop();
    t_sim.end();
}

int main(int argc, char *argv[])
{
    t_main.start();

    // igl::readOBJ(proj_dir_path + "/data/models/cloth.obj", pos_vis, tri);
    // num_particles = pos_vis.rows();
    num_particles = NV;
    printf("num_particles = %d\n", num_particles);

    run_simulation();

    // copy_pos_to_pos_vis();
    t_main.end("", "s");
}