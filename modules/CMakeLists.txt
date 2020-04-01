cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

add_executable("Example" "Main.cpp")
target_link_libraries("Example" PRIVATE Foo::Foo L/home/ioifrim/fSim++/dl-inference-module/modules/all/  -ltensorflow)
