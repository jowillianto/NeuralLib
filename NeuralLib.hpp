//Define Standard Libraries
#ifndef STD_IOSTREAM
#include <iostream>
#define STD_IOSTREAM
#endif
#ifndef STD_VECTOR
#include <vector>
#define STD_VECTOR
#endif
#ifndef CMATH
#include <cmath>
#define CMATH
#endif
#ifndef STD_TIME
#include <time.h>
#define STD_TIME
#endif
namespace nl{
    #ifndef LOCAL_NEURALNETWORK
    #define LOCAL_NEURALNETWORK
    #include "NeuralNetwork.hpp"
    #endif
    #ifndef LOCAL_TENSOR
    #define LOCAL_TENSOR
    #include "Tensor.hpp"
    #endif
}