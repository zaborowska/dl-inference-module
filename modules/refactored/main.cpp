#include "../../include/Model.h"
#include "../../include/Tensor.h"
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
    void initialization();
    auto generation();

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


void DLInference::initialization() {

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


auto DLInference::generation() {

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
    auto result = generatedEvent->get_data<float>();

    return result;
};


int main(int argc,char **argv) {
    
    DLInference session; 

    session.initialization();

    auto result = session.generation();

    for (auto it = result.begin() ; it != result.end(); ++it) 
        std::cout  << *it <<" , "<< std::endl;

    // Stream Event to File

    std::ofstream outFile("eventDCGAN.txt");

       for (const auto &e : result) outFile << e << "\n";
    return 0;
};



