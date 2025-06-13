cmake_minimum_required(VERSION 3.10)
project(onnx_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ONNX Runtime
find_package(onnxruntime REQUIRED)

# Add executable
add_executable(onnx_test main.cpp)

# Link libraries
target_link_libraries(onnx_test PRIVATE onnxruntime)

# Include directories
target_include_directories(onnx_test PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS}) 