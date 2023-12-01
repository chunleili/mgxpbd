cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_STANDARD 17)
project(mgxpbd)

set(use_CUDA ON CACHE BOOL "Use CUDA")

if(use_CUDA)
  find_package(CUDA REQUIRED)
  cuda_add_executable(main_cu main.cu)
else()




#------------------------------  targets ----------------------------- 
add_executable(main main.cpp)

target_include_directories(main PUBLIC src)

add_custom_command(
  TARGET main POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_BINARY_DIR}/$<CONFIGURATION> ${CMAKE_CURRENT_LIST_DIR}/bin
  )


# mmio
add_executable(mmread 
                extern/mmio/example_read.c 
                extern/mmio/mmio.c)

# --------------------------------------------------------------------------#
#                             external libraries                            #
# --------------------------------------------------------------------------#

# Although it is a header-only library, it still needs to use add_subdirectory, because some dependencies of libigl is managed by CMake.
add_subdirectory(extern/libigl) 
igl_include(glfw)
igl_include(imgui)
target_include_directories(main PUBLIC extern/libigl/include)
target_link_libraries(main PUBLIC 
  igl::glfw
  ## Other modules you could link to
  # igl::embree
  # igl::imgui
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

target_sources(main PRIVATE extern/mmio/mmio.c)
target_include_directories(main PUBLIC extern/mmio)

endif()