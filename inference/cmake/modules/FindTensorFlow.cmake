include(FindPackageHandleStandardArgs)

unset(TENSORFLOW_FOUND)

find_path(TensorFlow_INCLUDE_DIR NAMES tensorflow/c)

find_library(TensorFlow_LIBRARY NAMES tensorflow)

get_filename_component(TensorFlow_LIBRARY_DIR ${TensorFlow_LIBRARY} DIRECTORY)

find_package_handle_standard_args(TensorFlow DEFAULT_MSG TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY TensorFlow_LIBRARY_DIR)

if(TENSORFLOW_FOUND)
    set(TensorFlow_LIBRARY ${TensorFlow_LIBRARY})
    set(TensorFlow_INCLUDE_DIR ${TensorFlow_INCLUDE_DIR})
    set(TensorFlow_LIBRARY_DIR ${TensorFlow_LIBRARY_DIR})
endif()
