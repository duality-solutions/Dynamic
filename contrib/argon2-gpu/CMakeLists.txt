cmake_minimum_required(VERSION 3.5)

project(argon2-gpu CXX)
set(BINARY_INSTALL_DIR /usr/local/bin)
set(LIBRARY_INSTALL_DIR /usr/local/lib)
set(INCLUDE_INSTALL_DIR /usr/local/include)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

set(CUDA_FOUND FALSE)

if(NOT NO_CUDA)
    find_package(CUDA)
endif()

if(CUDA_FOUND)
    message("INFO: Using CUDA version ${CUDA_VERSION}")
    add_definitions(-DHAVE_CUDA=1)
else()
    message("INFO: Building without CUDA support")
    add_definitions(-DHAVE_CUDA=0)
endif()

if(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -std=c++11;-O3;--ptxas-options=-v;-arch sm_30;-lineinfo
    )
endif()

add_library(argon2-gpu SHARED
    src/argon2-gpu/common.cpp
    src/argon2-gpu/blake2b.cpp
)
target_include_directories(argon2-gpu INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(argon2-gpu PRIVATE
    include/
    src/argon2-gpu
)

if(CUDA_FOUND)
    cuda_add_library(argon2-cuda SHARED
        src/argon2-cuda/device.cpp
        src/argon2-cuda/global-context.cpp
        src/argon2-cuda/kernels.cu
        src/argon2-cuda/processing-unit.cpp
        src/argon2-cuda/program-context.cpp
    )
else()
    add_library(argon2-cuda SHARED
        src/argon2-gpu/empty.cpp
    )
endif()

target_include_directories(argon2-cuda PRIVATE
    include/
    src/argon2-cuda
)
target_include_directories(argon2-cuda INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(argon2-cuda argon2-gpu)

add_library(argon2-opencl SHARED
    src/argon2-opencl/device.cpp
    src/argon2-opencl/global-context.cpp
    src/argon2-opencl/kernel-loader.cpp
    src/argon2-opencl/kernel-runner.cpp
    src/argon2-opencl/processing-unit.cpp
    src/argon2-opencl/program-context.cpp
)
target_include_directories(argon2-opencl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(argon2-opencl PRIVATE
    include/argon2-opencl
    src/argon2-opencl
)
if (APPLE)
    target_link_libraries(argon2-opencl argon2-gpu "-framework OpenCL")
else()
    target_link_libraries(argon2-opencl argon2-gpu)
endif(APPLE)
install(
    TARGETS argon2-gpu argon2-opencl argon2-cuda
    DESTINATION ${LIBRARY_INSTALL_DIR}
)
install(FILES
    include/argon2-cuda/cuda-exception.h
    include/argon2-cuda/device.h
    include/argon2-cuda/global-context.h
    include/argon2-cuda/kernels.h
    include/argon2-cuda/processing-unit.h
    include/argon2-cuda/program-context.h
    DESTINATION ${INCLUDE_INSTALL_DIR}/argon2-cuda
)

install(FILES
    include/argon2-gpu/blake2b.h
    include/argon2-gpu/common.h
    DESTINATION ${INCLUDE_INSTALL_DIR}/argon2-gpu
)

install(FILES
    include/argon2-opencl/cl.hpp
    include/argon2-opencl/device.h
    include/argon2-opencl/global-context.h
    include/argon2-opencl/kernel-loader.h
    include/argon2-opencl/kernel-runner.h
    include/argon2-opencl/opencl.h
    include/argon2-opencl/processing-unit.h
    include/argon2-opencl/program-context.h
    DESTINATION ${INCLUDE_INSTALL_DIR}/argon2-opencl
)
