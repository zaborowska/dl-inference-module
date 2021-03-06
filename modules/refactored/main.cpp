#include "../../include/DLInference.h"
#include "../../include/Model.h"
#include "../../include/Tensor.h"
#include <algorithm>
#include <iterator>
#include <numeric>
#include <iomanip>
#include <vector>
#include <fstream>
#include <stdio.h>

int main(int argc,char **argv) {
    
    DLInference session; 

    session.init();

    auto result = session.generation();

    for (auto it = result.begin() ; it != result.end(); ++it) 
        std::cout  << *it <<" , "<< std::endl;

    // Stream Event to File

    std::ofstream outFile("eventDCGAN.txt");
       for (const auto e : result) outFile << e << "\n";
    return 0;

};

