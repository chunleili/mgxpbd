
enable_language(CUDA)
find_package(CUDAToolkit) 

# add_library(amgx STATIC IMPORTED)
# set_property(TARGET amgx PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/amgx.lib)
# target_include_directories(amgx INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(amgx_capi amgx_capi.c)

# include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

target_include_directories(amgx_capi PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

set(dyn_libs  ${CMAKE_CURRENT_SOURCE_DIR}/lib/amgxsh.lib CUDA::cudart_static CUDA::cublas)

target_link_libraries(amgx_capi ${dyn_libs} )

add_custom_command(
        TARGET amgx_capi  POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_LIST_DIR}/lib ${CMAKE_BINARY_DIR}/extern/amgx/$<CONFIGURATION>
        )

add_executable(convert convert.cpp)
target_include_directories(convert PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_link_libraries(convert ${dyn_libs} )

add_custom_command(
        TARGET convert  POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_LIST_DIR}/lib ${CMAKE_BINARY_DIR}/extern/amgx/$<CONFIGURATION>
        )