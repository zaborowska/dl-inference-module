#ifndef DL_INFERENCE_DLInference_H
#define DL_INFERENCE_DLInference_H

#include "Model.h"
#include "Tensor.h"
#include <algorithm>
#include <iterator>
#include <numeric>
#include <iomanip>
#include <vector>
#include <sstream> 
#include <fstream>
#include <stdio.h>
#include <functional>


class DLInference {

public:

    void init();
    std::vector<float> generation();

private:

    std::string modelType;
    int energyValue;
    std::string modelGraph;
    std::string modelRestore;
    std::string inputNode;
    std::string labelNode;
    std::string xiNode;
    std::string outputNode;

    int inputSize;
    int labelSize;
    int inputVecNumber;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> labelShape;

    std::string outFileName;

};



#endif //DL_INFERENCE_DLInference_H

