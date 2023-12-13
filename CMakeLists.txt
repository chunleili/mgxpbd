cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 17)
project(mgxpbd)

enable_language(CUDA)
add_executable(main src/main.cu)
target_include_directories(main PUBLIC include)
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(main PRIVATE Eigen3::Eigen)

find_package(libigl CONFIG REQUIRED)
target_link_libraries(main PRIVATE igl::igl_core igl_copyleft::igl_copyleft_core)
