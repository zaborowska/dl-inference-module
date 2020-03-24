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

int main(int argc,char **argv) {


    std::string modelType =  argv[1];
    std::string energyInput = argv[2];

    std::stringstream energy(energyInput);
    int energyValue = 0; 
    energy >> energyValue; 

    std::string modelGraph;
    std::string modelRestore;
    std::string inputNode;
    std::string labelNode;
    std::string outputNode;
    std::string outFileName;

    int inputSize;
    int labelSize;
    int inputVecNumber;

    if (modelType.compare("dcgan") == 0) {

        modelGraph = "../dcgan.pb";
        modelRestore = "../checkpoint/model.b64.ckpt";

        inputNode  = "z_input";
        labelNode  = "y_input";
        outputNode = "generator/gen_output";


        // Feed Data to Energy Label Tensor

        inputSize = 6400;
        labelSize = 640;

        inputVecNumber = 5;
        
        outFileName = "../eventDCGan.txt";

    }

    if (modelType.compare("cvae") == 0) {

        modelGraph = "../cvae.pb";
        modelRestore = "../checkpoint/progress-20-model.ckpt";

        inputNode  = "z_input";
        labelNode  = "y_input";
        outputNode = "decoder/x_decoder_mean_output";


        // Feed Data to Energy Label Tensor

        inputSize = 200;
        labelSize = 1000;

        inputVecNumber = 5;
        
        outFileName = "../eventCVAE.txt";

    }

    if (modelType.compare("ar") == 0) {

        modelGraph = "../graphPx.pb";
        modelRestore = "../checkpoint/params_PbWO4.ckpt";

        inputNode  = "model/input_cells";
        labelNode  = "model/input_labels";
        outputNode = "mul";


        // Feed Data to Energy Label Tensor

        inputSize = 165888;
        labelSize = 12;

        inputVecNumber = 1e-6;
        
        outFileName = "../eventAR.txt";

    }

        // Create the Requested Model

    Model m(modelGraph);
    m.restore(modelRestore);

        // Create Necesary Tensors

    auto inputData = new Tensor(m, inputNode);
    auto eventEnergy = new Tensor(m, labelNode);
    auto generatedEvent = new Tensor(m, outputNode);

    std::vector<float> inputVec(inputSize);
    std::vector<float> energies(labelSize);
    std::fill(inputVec.begin(), inputVec.end(), inputVecNumber);
    std::fill(energies.begin(), energies.end(), energyValue);

    if (modelType.compare("dcgan") == 0) {
        inputData->set_data(inputVec, {64,100});
        eventEnergy->set_data(energies, {64,10});

        m.run({inputData,eventEnergy}, generatedEvent);
    }

    if (modelType.compare("cvae") == 0) {

        inputData->set_data(inputVec, {100,2});
        eventEnergy->set_data(energies, {100,10});

        auto xinput  = new Tensor(m,"x_input");
        std::vector<float> xi(78400);
        std::fill(xi.begin(), xi.end(), 5);
        xinput->set_data(xi,{100,28,28,1});
    
        m.run({xinput,inputData,eventEnergy}, generatedEvent);
    }

    if (modelType.compare("ar") == 0) {

        inputData->set_data(inputVec, {12, 8,8, 24});
        eventEnergy->set_data(energies, {12});
    
        m.run({inputData,eventEnergy}, generatedEvent);
    }

        // Get Generated Event Tensor
    auto result = generatedEvent->get_data<float>();
     

    //      // Print result
    for (auto it = result.begin() ; it != result.end(); ++it) 
        std::cout  << *it <<" , "<< std::endl;
    //     std::cout << "2. its size: " << result.size() << '\n';    

        // Stream Event to File
    std::ofstream outFile(outFileName);
       for (const auto &e : result) outFile << e << "\n";
    
}



