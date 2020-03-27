#include <algorithm>
#include <iterator>
#include <numeric>
#include <iomanip>
#include <vector>
#include <sstream> 
#include <fstream>
#include <stdio.h>
#include <functional>
#include "./DLInference.h"

void DLInference::init() {

    //TODO Read from parameters file 
    modelType =  "dcgan";
   // std::string energyInput = argv[1];

    //std::stringstream energy(energyInput);
    energyValue = 0; 
    inputVecNumber = 5;
   // energy >> energyValue; 

    modelGraph = "../dcgan.pb" ;
    modelRestore = "../checkpoint/model.b64.ckpt";

    inputNode  = "z_input";
    labelNode  = "y_input";
    outputNode = "generator/gen_output";
    
    inputShape = {64,100};
    labelShape = {64,10};

    inputSize = std::accumulate(begin(inputShape), end(inputShape), 1, std::multiplies<>());
    labelSize = std::accumulate(begin(labelShape), end(labelShape), 1, std::multiplies<>());

    // outFileName = "./event" + modelType + std::to_string(energyValue) + ".txt";

};


std::vector<float> DLInference::generation() {

    // Create  Model
    Model m(modelGraph);
    m.restore(modelRestore);

    // Create Necesary Tensors

    auto inputData = new Tensor(m, inputNode);
    auto eventEnergy = new Tensor(m, labelNode);
    auto generatedEvent = new Tensor(m, outputNode);
    //auto xinput  = new Tensor(m,xiNode);

    // Feed Data to Tensors

    std::vector<float> inputVec(inputSize);
    std::vector<float> energies(labelSize);
    std::fill(inputVec.begin(), inputVec.end(), inputVecNumber);
    std::fill(energies.begin(), energies.end(), energyValue);

    inputData->set_data(inputVec, inputShape);
    eventEnergy->set_data(energies, labelShape);

    m.run({inputData,eventEnergy}, generatedEvent);

    // Get Generated Event Tensor
    std::vector<float> result = generatedEvent->get_data<float>();

    return result;
};

