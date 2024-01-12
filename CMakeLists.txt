cmake_minimum_required(VERSION 3.23)
set(CMAKE_CXX_STANDARD 17)
project(mgxpbd)

# if use cuda, comment out the following line
enable_language(CUDA)
add_executable(main src/main.cu)
target_include_directories(main PUBLIC include)
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# add_executable(main src/main.cpp)
# if use omp, comment out the following line
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(main PRIVATE Eigen3::Eigen)

find_package(libigl CONFIG REQUIRED)
target_link_libraries(main PRIVATE igl::igl_core igl_copyleft::igl_copyleft_core)


option(USE_VIEWER "Use viewer"	OFF)
if(USE_VIEWER)
    add_subdirectory(tools/viewer)
endif()