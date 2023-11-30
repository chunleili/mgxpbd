#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

std::string proj_dir_path;

Eigen::MatrixXd pos;  // vertex positions
Eigen::MatrixXi tri;  // triangle indices
Eigen::MatrixXi edge; // edge indices
Eigen::MatrixXf rest_len;

std::string get_directory_path(const std::string &filePath)
{
  size_t found = filePath.find_last_of("/\\");
  if (found != std::string::npos)
  {
    return filePath.substr(0, found);
  }
  return "";
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

int main(int argc, char *argv[])
{
  // find the project directory path
  std::string main_path = __FILE__;
  proj_dir_path = get_directory_path(main_path);
  std::cout << "Project directory path: " << proj_dir_path << std::endl;

  // Load a mesh
  igl::readOFF(proj_dir_path + "/data/models/bunny.OFF", pos, tri);
  auto pos_orig = pos; // keep original position for reset

  loadtxt(proj_dir_path + "/data/misc/edge.txt", edge);
  loadtxt(proj_dir_path + "/data/misc/rest_len.txt", rest_len);

  // Visualization
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(pos, tri);
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
      pos = pos_orig;
      viewer.data().set_vertices(pos);
      viewer.data().compute_normals();
      return true;
    }
  };

  viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &viewer) -> bool
  {
    glEnable(GL_CULL_FACE);
    if (viewer.core().is_animating)
    {
      // DoSubstep();
      for (int v = 0; v < pos.rows(); v++)
      {
        pos(v, 1) -= 0.001;
        // collide with y=0 plane
        if (pos(v, 1) < 0)
        {
          pos(v, 1) = 0.0;
          // ~ coefficient of restitution
        }
      }

      viewer.data().set_vertices(pos);
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