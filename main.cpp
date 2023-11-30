#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
#include <string>

std::string ProjDirPath;

Eigen::MatrixXd pos; //vertex positions
Eigen::MatrixXi tri; //triangle indices

std::string GetDirectoryPath(const std::string& filePath) {
    size_t found = filePath.find_last_of("/\\");
    if (found != std::string::npos) {
        return filePath.substr(0, found);
    }
    return "";
}


int main(int argc, char *argv[])
{
  // find the project directory path
  std::string mainCppPath = __FILE__;
  ProjDirPath = GetDirectoryPath(mainCppPath);
  std::cout << "Project directory path: " << ProjDirPath << std::endl;

  // Load a mesh
  igl::readOFF(ProjDirPath+"/data/models/bunny.OFF", pos, tri);
  auto posOrig = pos; //keep original position for reset

  // Visualization
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(pos, tri);
  // viewer.data().invert_normals = true;

  viewer.callback_key_pressed = 
    [&](igl::opengl::glfw::Viewer & viewer,unsigned int key,int mods)->bool
  {
    switch(key)
    {
      default: 
        return false;
      case ' ':
        viewer.core().is_animating = !viewer.core().is_animating;
        return true;
      case 'r':
        pos = posOrig;
        viewer.data().set_vertices(pos);
        viewer.data().compute_normals();
        return true;
    }
  };

  viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & viewer)->bool
  {
    glEnable(GL_CULL_FACE);
    if(viewer.core().is_animating)
    {
      // DoSubstep();
      for(int v = 0;v<pos.rows();v++)
      {
        pos(v,1) -= 0.001;
        // collide with y=0 plane
        if(pos(v,1) < 0)
        {
          pos(v,1) = 0.0;
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