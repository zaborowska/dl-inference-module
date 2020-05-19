# Library

Library that aids use of TensorFlow C API in the C++ projects that use CMake for the inference.
Models for inference are saved to files (from the training process) and read using this library.

## Requirements:

TensorFlow:
  - e.g. downloaded from https://www.tensorflow.org/install/lang_c and extracted to <TENSORFLOW_PATH>
  - from LCG releases on cvmfs, e.g. from LCG_96b 

## Installation


```
mkdir build
cd build
```

with downloaded C_API:
```
cmake .. -DCMAKE_PREFIX_PATH=<TENSORFLOW_PATH> -DCMAKE_INSTALL_PREFIX=../install
make install
```

with cvmfs (on lxplus = centos7):
```
source /cvmfs/sft.cern.ch/lcg/contrib/gcc/9.2.0/x86_64-centos7-gcc9-opt/setup.sh
cmake3 .. -DCMAKE_PREFIX_PATH=/cvmfs/sft.cern.ch/lcg/releases/LCG_96b/tensorflow/1.14.0/x86_64-centos7-gcc9-opt  -DCMAKE_INSTALL_PREFIX=../install
make install
```

with cvmfs (on lxplus8 = centos8):
```
source /cvmfs/sft.cern.ch/lcg/contrib/gcc/9.2.0/x86_64-centos7-gcc9-opt/setup.sh
/cvmfs/sft.cern.ch/lcg/contrib/CMake/3.14.2/Linux-x86_64/bin/cmake .. -DCMAKE_PREFIX_PATH=/cvmfs/sft.cern.ch/lcg/releases/LCG_96b/tensorflow/1.14.0/x86_64-centos7-gcc9-opt  -DCMAKE_INSTALL_PREFIX=../install
make install
```

Library will be installed to the directory specified to <CMAKE_INSTALL_PREFIX>, called later as <PATH_TO_THIS_LIBRARY_INSTALL_PATH>.
Header files, configuration file, and example networks will be installed there as well.

## Use this library in your project

In order to use this library in the project, add to its `CMakeLists.txt`:
```
list(APPEND CMAKE_MODULE_PATH ${CMAKE_PREFIX_PATH}/cmake)
find_package(InferenceLib)
[...]
add_executable([...])
target_link_libraries(<YOUR_PROJECT> InferenceLib)
```

and run cmake as before, adding `-DCMAKE_PREFIX_PATH=<PATH_TO_THIS_LIBRARY_INSTALL_PATH>`
