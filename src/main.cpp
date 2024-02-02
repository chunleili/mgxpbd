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

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

#ifdef USE_LIBIGL
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#endif

#ifdef USE_CUDA
#include "cuda.cuh"
#endif

using namespace std;
using Eigen::Map;
using Eigen::Vector3f;
using Eigen::VectorXf;
using SpMat = Eigen::SparseMatrix<float>;
using Triplet = Eigen::Triplet<float>;

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
string solver_type = "AMG";//"AMG", "JACOBI", "GS"
bool should_load_adjacent_edge=true;
float dual_residual[end_frame+1]={0.0};
bool use_off_diag = false;

// typedefs
using Vec3f = Eigen::Vector3f;
using Vec2i = std::array<int, 2>; // using Vec2i = Eigen::Vector2i;
using Vec3i = std::array<int, 3>; // using Vec3i = Eigen::Vector3i;
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
// Field1f b(M);
// Field1f dLambda(M);
FieldXi v2e; // vertex to edges
FieldXi adjacent_edge; //give a edge idx, get all its neighbor edges
FieldXi edge_abi; //(a,b,i): vertex a, vertex b, edge i. (a<b)
// vector<vector<Vec3i>> adjacent_edge_abc; //give a edge idx, get all its neighbor edges in a b c order, where a is the shared vertex
FieldXi adjacent_edge_abc; //give a edge idx, get all its neighbor edges in a b c order, where a is the shared vertex
Field1i num_adjacent_edge; //give a edge idx, get the number of its neighbor edges

// we have to use pos_vis for visualization because libigl uses Eigen::MatrixXd
Eigen::MatrixXd pos_vis;
Eigen::MatrixXi tri_vis;

Eigen::SparseMatrix<float> R, P;
Eigen::SparseMatrix<float> M_inv(3 * NV, 3 * NV);
Eigen::SparseMatrix<float> ALPHA(M, M);
Eigen::SparseMatrix<float> A(M, M);
Eigen::SparseMatrix<float> G(M, 3 * NV);
Eigen::VectorXf b(M);
Eigen::VectorXf dLambda(M);

// manually sparse matrix
struct MySpMat
{
    int num_rows;
    int num_cols;
    int num_nonzeros;
    
    vector<int> csr_row_start;
    vector<int> csr_col_idx;
    vector<float> csr_val;

    vector<int> coo_i;
    vector<int> coo_j;
    vector<float> coo_v;
};
MySpMat my_A;

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

template<typename T = Eigen::VectorXf>
void saveVector(T& d, string filename = "vec")
{
    Eigen::saveMarket(d, filename);
}

template<typename T = SpMat>
 void saveMatrix(T& d, string filename = "mat")
 {
     Eigen::saveMarket(d, filename);
 }

template<typename T=Field1i>
void savetxt(string filename, T &field)
{
    ofstream myfile;
    myfile.open(filename);
    for(auto &i:field)
    {
        myfile << i << '\n';
    }
    myfile.close();
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

//free from libigl
void write_obj_my_impl(std::string out_mesh_name, Field3f &pos, Eigen::MatrixXi &tri_vis)
{
    std::ofstream myfile;
    myfile.open(out_mesh_name);
    for (int i = 0; i < pos.size(); i++)
    {
        myfile << "v " << pos[i][0] << " " << pos[i][1] << " " << pos[i][2] << "\n";
    }
    for (int i = 0; i < tri_vis.rows(); i++)
    {
        myfile << "f " << tri_vis(i, 0) + 1 << " " << tri_vis(i, 1) + 1 << " " << tri_vis(i, 2) + 1 << "\n";
    }
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

    #if USE_LIBIGL
    copy_pos_to_pos_vis();
    igl::writeOBJ(out_mesh_name, pos_vis, tri_vis);
    #else
    write_obj_my_impl(out_mesh_name, pos, tri_vis);
    #endif

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

void clean_results_dir()
{
    std::string path = proj_dir_path + "/results/";

    for (const auto& entry : std::filesystem::directory_iterator(path)) 
    {
        if (entry.path().extension() == ".obj")
        {
            std::filesystem::remove(entry.path());
        }
    }

    std::cout<<"clean results dir done!"<<endl;
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


// input an edge index, output all its adjacent edges in a b c order, 
// where a is the shared vertex, b is another vertex on this edge, c is another vertex on the adjacent edge
//
// Example: input 0,  output: 1 0 2 0 1 257...
// where 1 0 2 is the first adjacent edge pair, 0 1 257 is the second adjacent edge pair, etc.
//
// Usage:
// for (int i = 0; i < NE; i++)
//     for (int j = 0; j < num_adjacent_edge; j++)
//     {
//         int a = adjacent_edge_abc[i][j * 3];
//         int b = adjacent_edge_abc[i][j * 3 + 1];
//         int c = adjacent_edge_abc[i][j * 3 + 2];
//     }
void init_adjacent_edge_abc()
{
    adjacent_edge_abc.resize(NE);
    // #pragma omp parallel for
    for (int i = 0; i < NE; i++)
    {   
        int ii0 = edge[i][0];
        int ii1 = edge[i][1];

        vector<int> adj = adjacent_edge[i];
        int num_adj = adj.size();
        adjacent_edge_abc[i].reserve(num_adj*3);
        for (int j = 0; j < num_adj; j++)
        {
            int ia = adj[j];
            if(ia==i)
            {
                printf("%d self!\n",ia);
                continue;
            }

            int jj0 = edge[ia][0];
            int jj1 = edge[ia][1];

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
                printf("%d no shared vertex!\n",ia);
                continue;
            }
            
            adjacent_edge_abc[i].push_back(a);
            adjacent_edge_abc[i].push_back(b);
            adjacent_edge_abc[i].push_back(c);
        }

    }
}

void init_num_adjacent_edge()
{
    num_adjacent_edge.resize(NE);
    for (int i = 0; i < NE; i++)
    {
        num_adjacent_edge[i] = adjacent_edge[i].size();
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
    std::vector<Triplet> inv_mass_3(3 * NV);
    for (int i = 0; i < 3 * NV; i++)
    {
        inv_mass_3[i] = Triplet(i, i, inv_mass[int(i / 3)]);
    }
    M_inv.setFromTriplets(inv_mass_3.begin(), inv_mass_3.end());
    M_inv.makeCompressed();
}

void fill_ALPHA()
{
    std::vector<Triplet> alpha_(NE);
    for (int i = 0; i < NE; i++)
    {
        alpha_[i] = Triplet(i, i, alpha);
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
    std::vector<Triplet> gradC_triplets;
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
                gradC_triplets.push_back(Triplet(j, 3 * pid + d, gradC[j][p][d]));
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


void init_A_pattern()
{
    A.reserve(Eigen::VectorXf::Constant(M, 15));

    // #pragma omp parallel for
    for (int i = 0; i < NE; i++)
    {   
        vector<int> adj = adjacent_edge[i];
        float diag = inv_mass[edge[i][0]] + inv_mass[edge[i][1]] + alpha;
        A.coeffRef(i,i) = diag;

        if(use_off_diag)
        {
            for(int j=0; j < adj.size(); j++)
            {
                int ia = adj[j];
                float off_diag = 0.0;
                A.coeffRef(i,ia) = off_diag;
            }
        }
    }
    A.makeCompressed();
}


void init_A_pattern_and_csr_coo_arr()
{
    std::vector<Triplet> val;
    val.reserve(15*NE);

    A.reserve(Eigen::VectorXf::Constant(M, 15));

    int cnt_nonzero = 0;
    my_A.csr_row_start.push_back(cnt_nonzero);
    // #pragma omp parallel for
    for (int i = 0; i < NE; i++)
    {   
        vector<int> adj = adjacent_edge[i];

        //set diagonal
        float diag = inv_mass[edge[i][0]] + inv_mass[edge[i][1]] + alpha;
        A.coeffRef(i,i) = diag;
        
        cnt_nonzero++;

        my_A.coo_i.push_back(i);
        my_A.coo_j.push_back(i);
        my_A.coo_v.push_back(diag);

        my_A.csr_col_idx.push_back(i);
        my_A.csr_val.push_back(diag);

        for(int j=0; j < adj.size(); j++)
        {
            int ia = adj[j];

            float off_diag = 0.0;
            A.coeffRef(i,ia) = off_diag;

            cnt_nonzero++;

            my_A.coo_i.push_back(i);
            my_A.coo_j.push_back(ia);
            my_A.coo_v.push_back(off_diag);

            my_A.csr_col_idx.push_back(ia);
            my_A.csr_val.push_back(off_diag);
        }

        my_A.csr_row_start.push_back(cnt_nonzero);
    }
    A.makeCompressed();
}


// // the bug of fill_A_cuda is not fix yet
// void fill_A_cuda()
// {
//     A.reserve(Eigen::VectorXf::Constant(M, 15));

//     thrust::device_vector<Vec2i> d_edge(edge.begin(), edge.end());
//     thrust::device_vector<Vec3f> d_pos(pos.begin(), pos.end());
//     thrust::device_vector<float> d_inv_mass(inv_mass.begin(), inv_mass.end());
    
//     for(int i=0;i<5;i++)
//     {
//         cout<<"d_edge["<<i<<"] = "<<static_cast<Vec2i>(d_edge[i])[1]<<endl;
//     }

//     // add one to each vertex
//     parallel_for<<<NE / 256, 256>>>(NE, 
//     [
//         edge = d_edge.data(),
//         inv_mass = d_inv_mass.data(),
//         alpha = alpha
//     ] __device__ (int i) 
//     {
//         //fill diagonal:m1 + m2 + alpha
//         int ii0 = static_cast<Vec2i>(edge[i])[0];
//         int ii1 = static_cast<Vec2i>(edge[i])[1];
//         // if (i==0)
//         //     printf("ii0 = %d, ii1 = %d\n",ii0,ii1);
//         float invM0 = inv_mass[ii0];
//         float invM1 = inv_mass[ii1];
//         float diag = (invM0 + invM1 + alpha);
//         // A.insert(i,i) = diag;

//         // //fill off-diagonal: m_a*dot(g_ab,g_ab)
//         // vector<int> adj = adjacent_edge[i];
//         // for (int j = 0; j < adj.size(); j++)
//         // {
//         //     int ia = adj[j];
//         //     if(ia==i)
//         //     {
//         //         printf("%d self!\n",ia);
//         //         continue;
//         //     }

//         //     int jj0 = edge[ia][0];
//         //     int jj1 = edge[ia][1];

//         //     // a is shared vertex 
//         //     // a-b is the first edge, a-c is the second edge
//         //     int a=-1,b=-1,c=-1;
//         //     if(ii0==jj0)
//         //     {
//         //         a=ii0;
//         //         b=ii1;
//         //         c=jj1;
//         //     }
//         //     else if(ii0==jj1)
//         //     {
//         //         a=ii0;
//         //         b=ii1;
//         //         c=jj0;
//         //     }
//         //     else if(ii1==jj0)
//         //     {
//         //         a=ii1;
//         //         b=ii0;
//         //         c=jj1;
//         //     }
//         //     else if(ii1==jj1)
//         //     {
//         //         a=ii1;
//         //         b=ii0;
//         //         c=jj0;
//         //     }
//         //     else
//         //     {
//         //         printf("%d no shared vertex!\n",ia);
//         //         continue;
//         //     }
            
            
//         //     // m_a*dot(g_ab,g_ab)
//         //     Vec3f g_ab = normalize(pos[a] - pos[b]);
//         //     Vec3f g_ac = normalize(pos[a] - pos[c]);
//         //     float off_diag = inv_mass[a] * dot(g_ab, g_ac);

//         //     A.insert(i,ia) = off_diag;
//         // }
//     });
//     checkCudaErrors(cudaDeviceSynchronize());
  
//     A.makeCompressed();
// }


// legacy code without warm start(init_A_pattern)
void fill_A_no_warm_start()
{
    std::vector<Triplet> val;
    val.reserve(15*NE);

    A.reserve(Eigen::VectorXf::Constant(M, 15));

    // #pragma omp parallel for
    for (int i = 0; i < NE; i++)
    {   
        //fill diagonal:m1 + m2 + alpha
        int ii0 = edge[i][0];
        int ii1 = edge[i][1];
        float invM0 = inv_mass[ii0];
        float invM1 = inv_mass[ii1];
        float diag = (invM0 + invM1 + alpha);
        val.push_back(Triplet(i, i, diag));
        // A.insert(i,i) = diag;
        // A.coeffRef(i,i) = diag;

        //fill off-diagonal: m_a*dot(g_ab,g_ab)
        vector<int> adj = adjacent_edge[i];
        for (int j = 0; j < adj.size(); j++)
        {
            int ia = adj[j];
            if(ia==i)
            {
                printf("%d self!\n",ia);
                continue;
            }

            int jj0 = edge[ia][0];
            int jj1 = edge[ia][1];

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
                printf("%d no shared vertex!\n",ia);
                continue;
            }
            
            
            // m_a*dot(g_ab,g_ab)
            Vec3f g_ab = normalize(pos[a] - pos[b]);
            Vec3f g_ac = normalize(pos[a] - pos[c]);
            float off_diag = inv_mass[a] * dot(g_ab, g_ac);

            val.push_back(Triplet(i, ia, off_diag));
            // A.insert(i,ia) = off_diag;
            // A.coeffRef(i,ia) = off_diag;
        }

    }
    A.setFromTriplets(val.begin(), val.end());
    A.makeCompressed();
}


//with warm start(init_A_pattern)
void fill_A()
{
    // #pragma omp parallel for
    for (int i = 0; i < NE; i++)
    {   
        //fill off-diagonal: m_a*dot(g_ab,g_ab)
        vector<int> adj = adjacent_edge[i];
        for (int j = 0; j < adj.size(); j++)
        {
            int ia = adj[j];
            int a = adjacent_edge_abc[i][j * 3];
            int b = adjacent_edge_abc[i][j * 3 + 1];
            int c = adjacent_edge_abc[i][j * 3 + 2];
            
            // m_a*dot(g_ab,g_ab)
            Vec3f g_ab = normalize(pos[a] - pos[b]);
            Vec3f g_ac = normalize(pos[a] - pos[c]);
            float off_diag = inv_mass[a] * dot(g_ab, g_ac);

            A.coeffRef(i,ia) = off_diag;
        }
    }
    // A.makeCompressed();
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

// An easy-to-use wrapper for gauss_seidel
void easy_gauss_seidel(const SpMat &A_=A, const VectorXf &b_=b, VectorXf &x_=dLambda)
{
    int max_GS_iter = 1;
    std::fill(x_.begin(), x_.end(), 0.0);
    for (int GS_iter = 0; GS_iter < max_GS_iter; GS_iter++)
    {
        gauss_seidel<int, float, float>(A_.outerIndexPtr(), A_.outerSize(),
                                        A_.innerIndexPtr(), A_.innerSize(), A_.valuePtr(), A_.nonZeros(),
                                        x_.data(), x_.size(), b_.data(), b_.size(), 0, b_.size(), 1);
    }
}

/*
 * Perform one iteration of Jacobi relaxation on the linear
 * system Ax = b, where A is stored in CSR format and x and b
 * are column vectors.  Damping is controlled by the omega
 * parameter.
 *
 * Refer to gauss_seidel for additional information regarding
 * row_start, row_stop, and row_step.
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
 * temp, array
 *     temporary vector the same size as x
 * row_start : int
 *     beginning of the sweep
 * row_stop : int
 *     end of the sweep (i.e. one past the last unknown)
 * row_step : int
 *     stride used during the sweep (may be negative)
 * omega : float
 *     damping parameter
 *
 * Returns
 * -------
 * Nothing, x will be modified inplace
 * 
 * https://github.com/pyamg/pyamg/blob/0431f825d7e6683c208cad20572e92fc0ef230c1/pyamg/amg_core/relaxation.h#L232
 *
 */
template<class I, class T, class F>
void jacobi(const I Ap[], const int Ap_size,
            const I Aj[], const int Aj_size,
            const T Ax[], const int Ax_size,
                  T  x[], const int  x_size,
            const T  b[], const int  b_size,
            const I row_start,
            const I row_stop,
            const I row_step,
            const T omega)
{
    T one = 1.0;

    std::vector<T> temp(x_size);

    for(I i = row_start; i != row_stop; i += row_step) {
        temp[i] = x[i];
    }

    for(I i = row_start; i != row_stop; i += row_step) {
        I start = Ap[i];
        I end   = Ap[i+1];
        T rsum = 0;
        T diag = 0;

        for(I jj = start; jj < end; jj++){
            I j = Aj[jj];
            if (i == j)
                diag  = Ax[jj];
            else
                rsum += Ax[jj]*temp[j];
        }

        if (diag != (F) 0.0){
            x[i] = (one - omega) * temp[i] + omega * ((b[i] - rsum)/diag);
        }
    }
}

// An easy-to-use wrapper for gauss_seidel
void easy_jacobi(const SpMat &A_=A, const VectorXf &b_=b, VectorXf &x_=dLambda, float omega_=omega)
{
    int max_jacobi_iter = 1;
    std::fill(x_.begin(), x_.end(), 0.0);
    for (int iter = 0; iter < max_jacobi_iter; iter++)
    {
        jacobi<int, float, float>(A_.outerIndexPtr(), A_.outerSize(),
                                  A_.innerIndexPtr(), A_.innerSize(), 
                                  A_.valuePtr(), A_.nonZeros(),
                                  x_.data(), x_.size(),
                                  b_.data(), b_.size(),
                                  0, b_.size(), 1, omega_);
    }
}


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


void fill_A_by_spmm()
{
    compute_C_and_gradC();
    fill_gradC_triplets();
    G.makeCompressed();
    A =  G * M_inv * G.transpose();
    fill_A_add_alpha();
}


void solve_amg(const SpMat& A_=A, const VectorXf& b_=b, VectorXf &x_=dLambda)
{
    float tol = 1e-3;
    int amg_max_iter = 1;

    SpMat A2 = R * A_ * P;

    Eigen::VectorXf residual = Eigen::VectorXf::Zero(M);
    Eigen::VectorXf coarse_b = Eigen::VectorXf::Zero(M);
    Eigen::VectorXf coarse_x = Eigen::VectorXf::Zero(P.cols());
    
    std::fill(x_.begin(), x_.end(), 0.0);

    float normb = b_.norm();
    if (normb == 0.0) 
        normb = 1.0;
    float normr = (b_ - A_ * x_).norm();
    for(int iter=0; iter < amg_max_iter && normr >= tol * normb; iter++)
    {
        residual = b_ - A_ * x_;
        coarse_b = R * residual; // restriction
        coarse_x.setZero();
        easy_gauss_seidel(A2, coarse_b, coarse_x); //coarse solve
        x_ += P * coarse_x; // prolongation
        // easy_gauss_seidel(A_, b_, x_); // smooth
        normr = (b_ - A_ * x_).norm();
    }
}

void substep_all_solver()
{
    semi_euler();
    reset_lagrangian();
    for (int iter = 0; iter <= max_iter; iter++)
    {
        t_iter.start();
        // printf("iter = %d ", iter);

        // assemble A and b
        if(use_off_diag)
        {
            fill_A();
        }

        update_constraints();
        fill_b();   //-C-alpha*lagrangian

        // solve Ax=b
        if (solver_type == "GS")
        {
            easy_gauss_seidel();
        }
        else if (solver_type == "AMG")
        {
            solve_amg();
        }
        else if (solver_type == "JACOBI")
        {
            easy_jacobi();
        }

        transfer_back_to_pos_mfree();

        calc_dual_residual(iter);
        // printf(" dual_residual = %f\n", dual_residual[iter]);

        // t_iter.end();
    }
    update_vel();
}

void main_loop()
{
    for (frame_num = 0; frame_num <= end_frame; frame_num++)
    {
        printf("\n\n----frame_num:%d----\n", frame_num);

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
    pos.resize(num_particles, Vec3f(0.0, 0.0, 0.0));
    edge.resize(NE);
    rest_len.resize(NE, 0.0);
    vel.resize(num_particles, Vec3f(0.0, 0.0, 0.0));
    inv_mass.resize(num_particles, 0.0);
    lagrangian.resize(NE, 0.0);
    constraints.resize(NE, 0.0);
    pos_mid.resize(num_particles, Vec3f(0.0, 0.0, 0.0));
    acc_pos.resize(num_particles, Vec3f(0.0, 0.0, 0.0));
    old_pos.resize(num_particles, Vec3f(0.0, 0.0, 0.0));
    tri.resize(3 * NT, 0);
    gradC.resize(NE, array<Vec3f, 2>{Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 0.0)});

    dLambda.resize(M);
    b.resize(M);

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
    init_num_adjacent_edge();
    init_adjacent_edge_abc();
    init_A_pattern();
    // savetxt("adjacent_edge.txt", adjacent_edge);
    // savetxt("adjacent_edge_abc.txt", adjacent_edge_abc);
    // savetxt("num_adjacent_edge.txt", num_adjacent_edge);

    //save A for generating R and P
    std::cout<<"save A for generating R and P\n";
    if(use_off_diag == false)
    {
        use_off_diag = true;
        init_A_pattern();
        fill_A();//fill off-diagal: m_a*dot(g_ab,g_ac)
        use_off_diag = false;
    }
    saveMatrix(A, proj_dir_path+"/data/misc/A.0.mtx");
    std::cout<<"A: "<<A.rows()<<"x"<<A.cols()<<std::endl;
    std::cout<<"A.nnz: "<<A.nonZeros()<<std::endl;
    saveMatrix(A, proj_dir_path+"/data/misc/A.0.mtx");
    std::cout<<"save A done\n";
    
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
    #ifdef USE_CUDA
    test_cuda();
    #endif
    
    t_main.start();

    clean_results_dir();

    num_particles = NV;
    printf("num_particles = %d\n", num_particles);

    run_simulation();
    
    t_main.end("", "s");
}