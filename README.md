# multigrid xpbd

## 编译
只试验了一种环境: VS2022 Widows 10

## 闲话
可视化依赖于libigl, 可以去extern/libigl/tutorials查看用法。

较大模型github无法上传, 请放在largeModels下面，不要污染了git仓库。

ProjDirPath 是全局变量，获取到的是项目的根目录，它依赖于main.cpp的位置，因此不要随意移动main.cpp的位置。

增加新的cpp文件放在src下面，然后在CMakeLists.txt中添加即可。例如要增加another.cpp
```cmake
add_executable(main main.cpp src/another.cpp)
```

增加libigl的一些第三方库，可以酌情注释掉下面的行。
```cmake
target_link_libraries(main PUBLIC 
  igl::glfw
  ## Other modules you could link to
  # igl::embree
  igl::imgui
  igl::opengl
  # igl::stb
  # igl::predicates
  # igl::xml
  # igl_copyleft::cgal
  # igl_copyleft::comiso
  # igl_copyleft::core
  # igl_copyleft::cork
  # igl_copyleft::tetgen
  # igl_restricted::matlab
  # igl_restricted::mosek
  # igl_restricted::triangle
  )
```