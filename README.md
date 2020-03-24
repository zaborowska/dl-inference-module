# Deep Learning Inference for Fast Simulation Applications 

C++ Inference module for Generative TensorFlow Models 


## How To Run Inference

Download the Tensorflow C API (https://www.tensorflow.org/install/lang_c) and extract its `./lib/` contents to `./module/all/`


Modify the `target_link_libraries` in `./module/ensemble/CMakeLists.txt` according to your path 

Moreover, you can run inference for your choice of model and energy input:

```sh
cd module/ensemble/
mkdir build
cd build
cmake ..
make .
./dlinf modelChoice energyValue 
```

where `modelChoice` can be either `dcgan` , `cvae`, `ar`




