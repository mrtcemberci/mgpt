cmake_minimum_required(VERSION 3.31)
project(mgpt)

set(CMAKE_CXX_STANDARD 20)

option(USE_CUDA "Build with CUDA support" ON)

include_directories(src/main src/main/utils src/main/neuralnetwork src/main/neuralnetwork/layers src/main/neuralnetwork/optimisers)

set(TENSOR_SOURCES src/main/utils/Tensor.cpp src/main/utils/cuda_ops.cu src/main/utils/Scratchpad.cpp)

if(USE_CUDA)
    if(NOT DEFINED CUDAToolkit_ROOT AND EXISTS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3")
        set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3")
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc.exe")
    endif()
    if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe")
        set(CMAKE_CUDA_HOST_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe")
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
    endif()
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    link_directories("${CUDAToolkit_ROOT}/lib/x64")
    link_libraries(cublas cudart)
    set_source_files_properties(src/main/utils/cuda_ops.cu PROPERTIES LANGUAGE CUDA)
else()
    set_source_files_properties(src/main/utils/cuda_ops.cu PROPERTIES LANGUAGE CXX)
endif()

enable_testing()

add_executable(mgpt src/main/main.cpp src/main/utils/Tokenizer.cpp
        ${TENSOR_SOURCES} src/main/utils/Tensor.h src/main/utils/cuda_ops.h src/main/neuralnetwork/layers/Layer.h
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
        src/main/neuralnetwork/layers/MultiHeadAttentionLayer.cpp
        src/main/neuralnetwork/layers/MultiHeadAttentionLayer.h
        src/main/neuralnetwork/layers/TransformerBlock.cpp
        src/main/neuralnetwork/layers/TransformerBlock.h
        src/main/neuralnetwork/GPT.cpp
        src/main/neuralnetwork/GPT.h
        src/main/neuralnetwork/optimisers/Optimiser.h
        src/main/neuralnetwork/optimisers/SGDOptimiser.h
        src/main/neuralnetwork/optimisers/AdamWOptimiser.h)
add_executable(utils_tests src/test/utilsTest/tokenizerTests.cpp src/main/utils/Tokenizer.cpp)
add_executable(scratchpad_tests src/test/utilsTest/scratchpadTests.cpp ${TENSOR_SOURCES})
add_executable(tensor_tests src/test/utilsTest/tensorTests.cpp ${TENSOR_SOURCES})
add_executable(mlp_tests src/test/networkTest/mlpTests.cpp ${TENSOR_SOURCES} src/main/neuralnetwork/layers/LinearLayer.cpp src/main/neuralnetwork/layers/GELULayer.cpp)
add_executable(norm_loss_tests src/test/networkTest/normAndLossTests.cpp ${TENSOR_SOURCES} src/main/neuralnetwork/layers/LayerNormLayer.cpp src/main/neuralnetwork/layers/CrossEntropyLossLayer.cpp)
add_executable(embedding_tests src/test/networkTest/embeddingTests.cpp ${TENSOR_SOURCES} src/main/neuralnetwork/layers/EmbeddingLayer.cpp)
add_executable(attention_tests src/test/networkTest/attentionTests.cpp ${TENSOR_SOURCES} src/main/neuralnetwork/layers/LinearLayer.cpp src/main/neuralnetwork/layers/SingleHeadAttentionLayer.cpp src/main/neuralnetwork/layers/MultiHeadAttentionLayer.cpp)
add_executable(optimiser_tests src/test/networkTest/optimiserTests.cpp ${TENSOR_SOURCES} src/main/neuralnetwork/layers/LinearLayer.cpp)
add_executable(gpt_tests src/test/networkTest/gptTests.cpp ${TENSOR_SOURCES} src/main/neuralnetwork/layers/LinearLayer.cpp src/main/neuralnetwork/layers/GELULayer.cpp src/main/neuralnetwork/layers/LayerNormLayer.cpp src/main/neuralnetwork/layers/CrossEntropyLossLayer.cpp src/main/neuralnetwork/layers/EmbeddingLayer.cpp src/main/neuralnetwork/layers/SingleHeadAttentionLayer.cpp src/main/neuralnetwork/layers/MultiHeadAttentionLayer.cpp src/main/neuralnetwork/layers/TransformerBlock.cpp src/main/neuralnetwork/GPT.cpp)
add_test(NAME utils_tests COMMAND utils_tests)
add_test(NAME scratchpad_tests COMMAND scratchpad_tests)
add_test(NAME tensor_tests COMMAND tensor_tests)
add_test(NAME mlp_tests COMMAND mlp_tests)
add_test(NAME norm_loss_tests COMMAND norm_loss_tests)
add_test(NAME embedding_tests COMMAND embedding_tests)
add_test(NAME attention_tests COMMAND attention_tests)
add_test(NAME optimiser_tests COMMAND optimiser_tests)
add_test(NAME gpt_tests COMMAND gpt_tests)
