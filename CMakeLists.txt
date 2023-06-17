cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
# modify to your own nvcc path, or delete it if ok
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")     # nvcc路径
project(CasualSelfAttention_Forward LANGUAGES CXX CUDA)

find_package(Python REQUIRED)
find_package(CUDA REQUIRED)

execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c
            "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE DCMAKE_PREFIX_PATH)

set(CMAKE_PREFIX_PATH "${DCMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# modify to your own python path, or delete it if ok
include_directories(/usr/include/python3.7)
include_directories(../include)

set(SRCS CasualSelfAttention_Foward_Warper.cpp CasualSelfAttention_Foward.cu)
add_library(CasualSelfAttention_Forward SHARED ${SRCS})

target_link_libraries(CasualSelfAttention_Forward "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")