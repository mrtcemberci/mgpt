cmake_minimum_required(VERSION 3.31)
project(mgpt)

set(CMAKE_CXX_STANDARD 20)

include_directories(src/main src/main/utils src/main/neuralnetwork/layers src/main/neuralnetwork/optimisers)

enable_testing()

add_executable(mgpt src/main/main.cpp src/main/utils/Tokenizer.cpp
        src/main/utils/Tensor.cpp
        src/main/utils/Tensor.h
        src/main/neuralnetwork/layers/Layer.h
        src/main/neuralnetwork/layers/LinearLayer.cpp
        src/main/neuralnetwork/layers/LinearLayer.h
        src/main/neuralnetwork/layers/GELULayer.cpp
        src/main/neuralnetwork/layers/LayerNormLayer.cpp
        src/main/neuralnetwork/layers/LayerNormLayer.h
        src/main/neuralnetwork/layers/CrossEntropyLossLayer.cpp
        src/main/neuralnetwork/layers/CrossEntropyLossLayer.h
        src/main/neuralnetwork/layers/EmbeddingLayer.cpp
        src/main/neuralnetwork/layers/EmbeddingLayer.h
        src/main/neuralnetwork/layers/SingleHeadAttentionLayer.cpp
        src/main/neuralnetwork/layers/SingleHeadAttentionLayer.h
        src/main/neuralnetwork/optimisers/Optimiser.h
        src/main/neuralnetwork/optimisers/SGDOptimiser.h
        src/main/neuralnetwork/optimisers/AdamWOptimiser.h)
add_executable(utils_tests src/test/utilsTest/tokenizerTests.cpp src/main/utils/Tokenizer.cpp)
add_executable(tensor_tests src/test/utilsTest/tensorTests.cpp src/main/utils/Tensor.cpp)
add_executable(mlp_tests src/test/networkTest/mlpTests.cpp src/main/utils/Tensor.cpp src/main/neuralnetwork/layers/LinearLayer.cpp src/main/neuralnetwork/layers/GELULayer.cpp)
add_executable(norm_loss_tests src/test/networkTest/normAndLossTests.cpp src/main/utils/Tensor.cpp src/main/neuralnetwork/layers/LayerNormLayer.cpp src/main/neuralnetwork/layers/CrossEntropyLossLayer.cpp)
add_executable(embedding_tests src/test/networkTest/embeddingTests.cpp src/main/utils/Tensor.cpp src/main/neuralnetwork/layers/EmbeddingLayer.cpp)
add_executable(attention_tests src/test/networkTest/attentionTests.cpp src/main/utils/Tensor.cpp src/main/neuralnetwork/layers/LinearLayer.cpp src/main/neuralnetwork/layers/SingleHeadAttentionLayer.cpp)
add_executable(optimiser_tests src/test/networkTest/optimiserTests.cpp src/main/utils/Tensor.cpp src/main/neuralnetwork/layers/LinearLayer.cpp)
add_test(NAME utils_tests COMMAND utils_tests)
add_test(NAME tensor_tests COMMAND tensor_tests)
add_test(NAME mlp_tests COMMAND mlp_tests)
add_test(NAME norm_loss_tests COMMAND norm_loss_tests)
add_test(NAME embedding_tests COMMAND embedding_tests)
add_test(NAME attention_tests COMMAND attention_tests)
add_test(NAME optimiser_tests COMMAND optimiser_tests)
