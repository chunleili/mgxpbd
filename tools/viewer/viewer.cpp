#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <thread>
#include <functional>

using namespace Eigen;
using namespace std;

using Vec3f = Eigen::Vector3f;
using Field3f = std::vector<Vec3f>;
using Vec2i = Eigen::Vector2i;
using Field2i = std::vector<Vec2i>;
using Field1f = std::vector<float>;

std::string proj_dir_path;
int wait_milliseconds = 1000;
bool is_test = false;
int start_frame = 1;
int end_frame = 2; //[start_frame, end_frame)
int paused = 0;

unsigned num_particles = 0;

Eigen::MatrixXd pos_vis; // vertex positions
Eigen::MatrixXd pos_orig;
Eigen::MatrixXi tri; // triangle indices

Field3f pos;

// copy pos to pos_vis for visualization
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

// copy pos_vis to pos
void copy_pos_vis_to_pos()
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
    pos[i][1] -= 0.01;
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

void load_external_animation()
{
  for (int i = start_frame; i < end_frame; i++)
  {
    if (i == start_frame)
    {
      pos_orig = pos_vis;
    }
    string filename = proj_dir_path + "/result/" + to_string(i) + ".obj";
    printf("Loading %s\n", filename.c_str());
    igl::readOBJ(filename, pos_vis, tri);
  }

  paused = 1;
  printf("Load external animation done\n");
}

void update_physics()
{
  if (is_test)
  {
    semi_euler();
    collision_response();
    copy_pos();
    return;
  }
  else
  {
    load_external_animation();
    return;
  }
}

/* -------------------------------------------------------------------------- */
/*                              utility functions                             */
/* -------------------------------------------------------------------------- */
void loadtxt(std::string filename, Field3f &M)
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

void loadtxt(std::string filename, Field2i &M)
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

std::string get_proj_dir_path()
{
  std::filesystem::path p(__FILE__);
  std::filesystem::path prj_path = p.parent_path().parent_path().parent_path();
  proj_dir_path = prj_path.string();

  std::cout << "Project directory path: " << proj_dir_path << std::endl;
  return proj_dir_path;
}
// this code run before main, in case of user forget to call get_proj_dir_path()
static string proj_dir_path_pre_get = get_proj_dir_path();
/* -------------------------------------------------------------------------- */
/*                          end of utility functions                          */
/* -------------------------------------------------------------------------- */

void visualization(std::function<void(void)> &update_physics_callback)
{
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
      paused = !paused;
      printf("paused %d\n", !viewer.core().is_animating);
      return true;
    case 'r':
      pos_vis = pos_orig;
      viewer.data().set_vertices(pos_vis);
      viewer.data().compute_normals();
      printf("reset\n");
      return true;
    case 'p': // speed up
      wait_milliseconds /= 2;
      printf("Faster! Wait %d ms every frame\n", wait_milliseconds);
      return true;
    case 'l': // speed down
      wait_milliseconds /= 2;
      printf("Slower! Wait %d ms every frame\n", wait_milliseconds);
      return true;
    }
  };

  viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &viewer) -> bool
  {
    glEnable(GL_CULL_FACE);
    if (viewer.core().is_animating && !paused)
    {
      // 定义等待的时间间隔，这里是 1 秒钟. 使程序休眠指定的时间
      std::chrono::milliseconds duration(wait_milliseconds);
      std::this_thread::sleep_for(duration);

      update_physics_callback();

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

void test()
{
  // Load a mesh
  igl::readOFF(proj_dir_path + "/data/models/bunny.OFF", pos_vis, tri);
  // igl::readOBJ(proj_dir_path + "/result/1.obj", pos_vis, tri);
  pos_orig = pos_vis; // keep original position for reset

  num_particles = pos_vis.rows();

  pos.resize(num_particles);

  copy_pos_vis_to_pos();

  // loadtxt(proj_dir_path + "/data/misc/edge.txt", edge);
  std::function<void(void)> func = update_physics;
  visualization(func);
}

int main(int argc, char *argv[])
{
  if (is_test)
  {
    printf("Test mode\n");
    test();
    return 0;
  }

  else
  {
    std::function<void(void)> func = update_physics;
    visualization(func);
  }
}