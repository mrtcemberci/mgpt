cmake_minimum_required(VERSION 3.31)
project(mgpt)

set(CMAKE_CXX_STANDARD 20)

include_directories(src/main src/main/utils)

enable_testing()

add_executable(mgpt src/main/main.cpp src/main/utils/Tokenizer.cpp)
add_executable(utils_tests src/test/utilsTest/tokenizerTests.cpp src/main/utils/Tokenizer.cpp)
add_test(NAME utils_tests COMMAND utils_tests)
