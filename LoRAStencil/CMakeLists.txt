cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build for")
endif()

project(LoRAStencil LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
enable_language(CXX CUDA)

option(LORASTENCIL_ENABLE_CHECKS "Enable CPU-vs-GPU correctness checks" OFF)

find_package(CUDAToolkit REQUIRED)

function(add_lorastencil_executable target)
        add_executable(${target} ${ARGN})
        set_target_properties(
                ${target}
                PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON)
        target_include_directories(${target} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --use_fast_math --ptxas-options=-v>)
        if(LORASTENCIL_ENABLE_CHECKS)
                target_compile_definitions(${target} PRIVATE CHECK_ERROR)
        endif()
endfunction()

add_lorastencil_executable(lorastencil_1d
        src/1d/main.cu
        src/1d/gpu_1r.cu
        src/1d/gpu_2r.cu
        src/1d/1d_utils.h)

add_lorastencil_executable(lorastencil_2d
        src/2d/main.cu
        src/2d/gpu.cu
        src/2d/2d_utils.h)

add_lorastencil_executable(lorastencil_3d
        src/3d/main.cu
        src/3d/gpu_box.cu
        src/3d/gpu_star.cu
        src/3d/3d_utils.h)
