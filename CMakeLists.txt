cmake_minimum_required(VERSION 3.10)
project(gpu_benchmark LANGUAGES CXX CUDA)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLAS_LIBRARY cublas ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

add_executable(
    ${PROJECT_NAME}
    benchmark.cu
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(${PROJECT_NAME}
    ${CUDART_LIBRARY}
    ${CUBLAS_LIBRARY}
)
