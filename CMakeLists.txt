cmake_minimum_required(VERSION 3.23)
set(CMAKE_CXX_STANDARD 17)
project(mgxpbd)

# if use cuda, comment out the following line
add_executable(main)

option(USE_CUDA "Use CUDA"	OFF)
if(USE_CUDA)
    enable_language(CUDA)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(main.cpp PROPERTIES LINKER_LANGUAGE CUDA)
    target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    target_sources(main PRIVATE src/cuda.cu)
    target_include_directories(main PUBLIC include/cuda)
    target_compile_definitions(main PUBLIC USE_CUDA)
endif()

target_sources(main PRIVATE src/main.cpp)

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(main PRIVATE Eigen3::Eigen)

option(USE_OPENMP "Use OpenMP"	OFF)
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

option(USE_LIBIGL "Use libigl"	OFF)
if(USE_LIBIGL)
    target_compile_definitions(main PUBLIC USE_LIBIGL)
    find_package(libigl CONFIG REQUIRED)
    target_link_libraries(main PRIVATE igl::igl_core igl_copyleft::igl_copyleft_core)
    target_compile_definitions(main PUBLIC USE_LIBIGL)
endif()

option(USE_VIEWER "Use viewer"	OFF)
if(USE_VIEWER)
    add_subdirectory(tools/viewer)
endif()