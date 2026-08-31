cmake_minimum_required(VERSION 3.16)

project(NBodySimulation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

add_executable(nbody
    source/main/main.cpp
    source/utils/fileUtilities.hpp
    source/utils/force.hpp
    source/utils/particle.hpp
    source/utils/profiling.hpp
    source/utils/quadtreeNode.hpp
    source/utils/simulationUtilities.hpp
    source/utils/treeUtilities.hpp
)

target_include_directories(nbody PRIVATE
    source/utils
)

target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX)
