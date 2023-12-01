#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "utils.h"

using namespace Eigen;
using namespace std;

using Vec3f = Eigen::Vector3f;
using Field3f = std::vector<Vec3f>;
using Vec2i = Eigen::Vector2i;
using Field2i = std::vector<Vec2i>;
using Field1f = std::vector<float>;

std::string proj_dir_path;

int N = 256;
int NV = (N + 1) * (N + 1);
int NT = 2 * N * N;
int NE = 2 * N * (N + 1) + N * N;
float h = 0.01;
int M = NE;
int new_M = int(NE / 100);
float compliance = 1.0e-8;
float alpha = compliance * (1.0 / h / h);

unsigned num_particles = 0;

Eigen::MatrixXd pos_vis;  // vertex positions
Eigen::MatrixXi tri;  // triangle indices

Field3f pos;
Field2i edge; 
Field3f vel;
Field1f inv_mass;
Field1f rest_len;
Field1f lagrangian;
Field1f constraints;
Field1f dLambda;
Field3f pos_mid;
Field3f acc_pos;
Field3f old_pos;

// Eigen::MatrixXi edge; // edge indices
// Eigen::MatrixXf rest_len;
// Eigen::MatrixXf lagrangian;
// Eigen::MatrixXf constraints;
// Eigen::MatrixXf dLambda;
// Eigen::MatrixXf pos_mid;
// Eigen::MatrixXf acc_pos;
// Eigen::MatrixXf old_pos;
// Eigen::MatrixXf vel;
// Eigen::MatrixXf inv_mass;
// Eigen::MatrixXf gradC;


// void copy_field(MatrixXd src, Field3f dst, bool reverse = false)
// {
//   for (int i = 0; i < src.rows(); i++)
//   {
//     for (int j = 0; j < src.cols(); j++)
//     {
//       if (!reverse)
//         dst(i, j) = src[i][j];
//       else
//         src[i][j] = dst(i, j);
//     }
//   }
// }

//copy pos to pos_vis for visualization
void copy_pos()
{
  for (unsigned i = 0; i < num_particles; i++)
  {
    for (int j = 0; j < 3; j++)
    {
        pos_vis(i, j) = pos[i][j];
    }
  }
}

//copy pos_vis to pos
void copy_pos_init()
{
  for (unsigned i = 0; i < num_particles; i++)
  {
    for (int j = 0; j < 3; j++)
    {
        pos[i][j] = pos_vis(i, j);
    }
  }
}

void semi_euler()
{
  Vec3f gravity(0.0, -0.01, 0.0);
  for (unsigned i = 0; i < num_particles; i++)
  {
    // if (inv_mass(i) != 0.0)
    {
      pos[i][1] -= 0.01;
      // vel.row(i) += h * gravity;
      // old_pos.row(i) = pos.row(i);
      // pos.row(i) += h * vel.row(i);
    }
  }
}


void collision_response()
{
    for (unsigned i = 0; i < num_particles; i++)
    {
      if (pos[i][1] < 0.0)
      {
        pos[i][1] = 0.0;
      }
    }
}


/* -------------------------------------------------------------------------- */
/*                              utility functions                             */
/* -------------------------------------------------------------------------- */
void loadtxt(std::string filename, Field3f& M)
{
  printf("Loading %s with Field3f\n", filename.c_str());
  std::ifstream inputFile(filename);
  std::string line;
  std::vector<float> values;
  unsigned int rows = 0;
  while (std::getline(inputFile, line))
  {
    std::istringstream iss(line);
    float val;
    while (iss >> val)
    {
      values.push_back(val);
    }
    rows++;
  }
  unsigned int num_per_row = 3;
  M.resize(rows);
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < num_per_row; j++)
    {
      M[i][j] = values[i * num_per_row + j];
    }
  }
}

void loadtxt(std::string filename, Field2i& M)
{
  printf("Loading %s with Field2i\n", filename.c_str());
  std::ifstream inputFile(filename);
  std::string line;
  std::vector<float> values;
  unsigned int rows = 0;
  while (std::getline(inputFile, line))
  {
    std::istringstream iss(line);
    int val;
    while (iss >> val)
    {
      values.push_back(val);
    }
    rows++;
  }
  unsigned int num_per_row = 2;
  M.resize(rows);
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < num_per_row; j++)
    {
      M[i][j] = values[i * num_per_row + j];
    }
  }
}

void loadtxt(std::string filename, Eigen::MatrixXi &M)
{
  printf("Loading %s with integer\n", filename.c_str());

  std::ifstream inputFile(filename);
  std::string line;
  std::vector<int> values;
  unsigned int rows = 0;
  unsigned int cols = 0;
  while (std::getline(inputFile, line))
  {
    std::istringstream iss(line);
    int val;
    while (iss >> val)
    {
      values.push_back(val);
      cols++;
    }
    rows++;
  }
  unsigned int numPerRow = static_cast<unsigned int>(cols / rows);
  M.resize(rows, numPerRow);
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < numPerRow; j++)
    {
      M(i, j) = values[i * numPerRow + j];
    }
  }
}

void loadtxt(std::string filename, Eigen::MatrixXf &M)
{
  printf("Loading %s with float\n", filename.c_str());
  std::ifstream inputFile(filename);
  std::string line;
  std::vector<float> values;
  unsigned int rows = 0;
  unsigned int cols = 0;
  while (std::getline(inputFile, line))
  {
    std::istringstream iss(line);
    float val;
    while (iss >> val)
    {
      values.push_back(val);
      cols++;
    }
    rows++;
  }
  unsigned int num_per_row = static_cast<unsigned int>(cols / rows);
  M.resize(rows, num_per_row);
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < num_per_row; j++)
    {
      M(i, j) = values[i * num_per_row + j];
    }
  }
}

std::string get_directory_path(const std::string &filePath)
{
  size_t found = filePath.find_last_of("/\\");
  if (found != std::string::npos)
  {
    return filePath.substr(0, found);
  }
  return "";
}
/* -------------------------------------------------------------------------- */
/*                          end of utility functions                          */
/* -------------------------------------------------------------------------- */

int main(int argc, char *argv[])
{
  // find the project directory path
  std::string main_path = __FILE__;
  proj_dir_path = get_directory_path(main_path);
  std::cout << "Project directory path: " << proj_dir_path << std::endl;

  // Load a mesh
  igl::readOFF(proj_dir_path + "/data/models/bunny.OFF", pos_vis, tri);
  auto pos_orig = pos_vis; // keep original position for reset

  num_particles = pos_vis.rows();

  pos.resize(num_particles);

  copy_pos_init();

  loadtxt(proj_dir_path + "/data/misc/edge.txt", edge);
  // loadtxt(proj_dir_path + "/data/misc/rest_len.txt", rest_len);

  // Visualization
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(pos_vis, tri);
  // viewer.data().invert_normals = true;

  viewer.callback_key_pressed =
      [&](igl::opengl::glfw::Viewer &viewer, unsigned int key, int mods) -> bool
  {
    switch (key)
    {
    default:
      return false;
    case ' ':
      viewer.core().is_animating = !viewer.core().is_animating;
      return true;
    case 'r':
      pos_vis = pos_orig;
      viewer.data().set_vertices(pos_vis);
      viewer.data().compute_normals();
      return true;
    }
  };

  viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &viewer) -> bool
  {
    glEnable(GL_CULL_FACE);
    if (viewer.core().is_animating)
    {
      semi_euler();

      collision_response();

      copy_pos();

      viewer.data().set_vertices(pos_vis);
      viewer.data().compute_normals();
    }
    return false;
  };

  viewer.data().show_lines = false;
  viewer.core().is_animating = true;
  viewer.core().animation_max_fps = 30.;
  viewer.data().set_face_based(true);

  viewer.launch();
}