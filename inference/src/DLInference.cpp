#include "DLInference.h"
#include "Model.h"
#include "Tensor.h"

void DLInference::init() {

    //TODO Read from parameters file 
    modelType =  "dcgan";
   // std::string energyInput = argv[1];

    //std::stringstream energy(energyInput);
    energyValue = 0; 
    inputVecNumber = 5;
   // energy >> energyValue; 

    modelGraph = "../cvae.pb" ;
    modelRestore = "../checkpoint/progress-20-model.ckpt";

    inputNode  = "z_input";
    labelNode  = "y_input";
    outputNode = "decoder/x_decoder_mean_output";
    
    inputShape = {100,2};
    labelShape = {100,10};

    inputSize = std::accumulate(begin(inputShape), end(inputShape), 1, std::multiplies<>());
    labelSize = std::accumulate(begin(labelShape), end(labelShape), 1, std::multiplies<>());

    // outFileName = "./event" + modelType + std::to_string(energyValue) + ".txt";

}


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

    auto xinput  = new Tensor(m,"x_input");
    std::vector<float> xi(78400);
    std::fill(xi.begin(), xi.end(), 5);
    xinput->set_data(xi,{100,28,28,1});
    m.run({xinput,inputData,eventEnergy}, generatedEvent);

    // Get Generated Event Tensor
    std::vector<float> result = generatedEvent->get_data<float>();

    return result;
}

void DLInference::SetModelGraph(const std::string &aModelGraph)
{
    modelGraph = aModelGraph;
}
void DLInference::SetModelRestore(const std::string &aModelRestore)
{
    modelRestore = aModelRestore;
}
