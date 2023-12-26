cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)
project(mgxpbd)

enable_language(CUDA)
add_executable(main src/main.cu)
target_include_directories(main PUBLIC include)
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(main PRIVATE Eigen3::Eigen)

# --------------------------------------------------------------------------#
#                                   libigl                                  #
# --------------------------------------------------------------------------#
# if use vcpkg
# find_package(libigl CONFIG REQUIRED)
# target_link_libraries(main PRIVATE igl::igl_core igl_copyleft::igl_copyleft_core)

# if use igl template
if(TARGET igl::core)
    return()
endif()
include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(libigl)

# Enable the target igl::glfw
# igl_include(glfw)
# Other modules you could enable
#igl_include(embree)
#igl_include(imgui)
#igl_include(opengl)
#igl_include(stb)
#igl_include(predicates)
#igl_include(xml)
#igl_include(copyleft cgal)
#igl_include(copyleft comiso)
igl_include(copyleft core)
#igl_include(copyleft cork)
#igl_include(copyleft tetgen)
#igl_include(restricted matlab)
#igl_include(restricted mosek)
#igl_include(restricted triangle)


# Link igl (and the glfw module) to your project
target_link_libraries(main PUBLIC 
#   igl::glfw
  ## Other modules you could link to
  # igl::embree
  # igl::imgui
  # igl::opengl
  # igl::stb
  # igl::predicates
  # igl::xml
  # igl_copyleft::cgal
  # igl_copyleft::comiso
  igl_copyleft::core
  # igl_copyleft::cork
  # igl_copyleft::tetgen
  # igl_restricted::matlab
  # igl_restricted::mosek
  # igl_restricted::triangle
  )
# -------------------------------------------------------------------------- #
#                                  end libigl                                #
# -------------------------------------------------------------------------- #


option(USE_VIEWER "Use viewer"	OFF)
if(USE_VIEWER)
    add_subdirectory(tools/viewer)
endif()