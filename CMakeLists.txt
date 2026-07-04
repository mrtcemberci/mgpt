cmake_minimum_required(VERSION 3.31)
project(mgpt)

set(CMAKE_CXX_STANDARD 20)

include_directories(src/main src/main/utils)

add_executable(mgpt src/main/main.cpp src/main/utils/DataLoader.cpp)
add_executable(utils_tests src/test/utilsTest/dataLoaderTests.cpp src/main/utils/DataLoader.cpp)
