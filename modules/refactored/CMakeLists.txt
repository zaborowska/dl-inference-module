cmake_minimum_required(VERSION 3.10)
project(dlinf)

set(CMAKE_CXX_STANDARD 17)
add_executable(dlinf main.cpp ../../src/Model.cpp ../../src/Tensor.cpp ../../src/DLInference.cpp)
target_include_directories(dlinf PRIVATE ../../include)
target_link_libraries (dlinf -L/home/ioifrim/fSim++/dl-inference-module/modules/all/  -ltensorflow)
