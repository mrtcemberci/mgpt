cmake_minimum_required(VERSION 3.31)
project(mgpt)

set(CMAKE_CXX_STANDARD 20)

include_directories(src/main src/main/utils src/main/neuralnetwork/layers)

enable_testing()

add_executable(mgpt src/main/main.cpp src/main/utils/Tokenizer.cpp
        src/main/utils/Tensor.cpp
        src/main/utils/Tensor.h
        src/main/neuralnetwork/layers/Layer.h
        src/main/neuralnetwork/layers/LinearLayer.cpp
        src/main/neuralnetwork/layers/LinearLayer.h
        src/main/neuralnetwork/layers/GELULayer.cpp
        src/main/neuralnetwork/layers/GELULayer.h)
add_executable(utils_tests src/test/utilsTest/tokenizerTests.cpp src/main/utils/Tokenizer.cpp)
add_executable(tensor_tests src/test/utilsTest/tensorTests.cpp src/main/utils/Tensor.cpp)
add_executable(mlp_tests src/test/networkTest/mlpTests.cpp src/main/utils/Tensor.cpp src/main/neuralnetwork/layers/LinearLayer.cpp src/main/neuralnetwork/layers/GELULayer.cpp)
add_test(NAME utils_tests COMMAND utils_tests)
add_test(NAME tensor_tests COMMAND tensor_tests)
add_test(NAME mlp_tests COMMAND mlp_tests)
