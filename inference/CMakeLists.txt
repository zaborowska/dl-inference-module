#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 3.10)
project(InferenceLib VERSION 0.1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -DDROP_CGAL ${CPP17FLAGS} -Wall -Wextra -Wpedantic -Wno-unused-variable -Wno-unused-parameter")

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(SEND_ERROR "In-source builds are not allowed.")
endif ()

#----------------------------------------------------------------------------
# Setup required TensorFlow dependency
#
find_package(TensorFlow REQUIRED)

#----------------------------------------------------------------------------
# Locate sources and headers for this project
#
include_directories(${PROJECT_SOURCE_DIR}/include/
                    ${TensorFlow_INCLUDE_DIR})
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.h)

#----------------------------------------------------------------------------
# Add the library, correctly setup the include directories, and link the library to the TensorFlow libraries
#
add_library(${PROJECT_NAME} SHARED ${sources})
target_include_directories(${PROJECT_NAME} PUBLIC ${TensorFlow_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include> PRIVATE src)
target_link_directories(${PROJECT_NAME} PUBLIC ${TensorFlow_LIBRARY_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC tensorflow tensorflow_framework)
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

#----------------------------------------------------------------------------
# Install the library, headers, config, and models to directories under CMAKE_INSTALL_PREFIX
#
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Config LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

install(EXPORT ${PROJECT_NAME}Config DESTINATION cmake)
export(TARGETS ${PROJECT_NAME} FILE ${PROJECT_NAME}Config.cmake)

install(FILES ../modules/dcgan.pb ../modules/cvae.pb ../modules/graphPx.pb DESTINATION networks)
install(FILES ../modules/checkpoint/model.b64.ckpt.data-00000-of-00001 ../modules/checkpoint/model.b64.ckpt.meta
              ../modules/checkpoint/model.b64.ckpt.index ../modules/checkpoint/progress-20-model.ckpt.data-00000-of-00001
              ../modules/checkpoint/progress-20-model.ckpt.index ../modules/checkpoint/progress-20-model.ckpt.meta
              DESTINATION networks)
