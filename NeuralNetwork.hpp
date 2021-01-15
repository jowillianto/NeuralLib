#ifndef LOCAL_TENSOR
#define LOCAL_TENSOR
#include "Tensor.hpp"
#endif

#ifndef CMATH
#define CMATH
#include <cmath>
#endif

//Template for Neural Network Declaration
class Model{
    public: 
        virtual Tensor forward(Tensor& Input) = 0;
};

//Template for all class (You can Declare a Sequential with this interface with a vector)
class NetworkFunc{
    public: 
        virtual Tensor forward(Tensor& Input) = 0;
};

//Standard NN Network 
class Linear : public NetworkFunc{
    public:
        Linear(const long& Input, const long& Output, const bool& Bias) : _Weight_({Input, Output}), _Bias_({Output, 1}){
            //Weigth initialized of size m x n : Input x Output. If there are 7 layers of Input, then the
            //Resulting Matrix will be 7 * Output
            _UseBias_   = Bias;
        }
        Tensor forward(Tensor& Input){
            Tensor returnTensor     = Input * _Weight_;
            if(_UseBias_){
                return _AddBias_(returnTensor);
            }
            else{
                return returnTensor;
            }
        }
        Tensor& weight(){
            return _Weight_;
        }
        Tensor& bias(){
            return _Bias_;
        }
    private:
        Tensor _Weight_;
        Tensor _Bias_;
        bool _UseBias_;

        //Adding all by the Bias
        Tensor _AddBias_(Tensor& Input){
            Tensor returnTensor({Input.size()});
            for(int i = 0; i < Input.size().height;i++){
                for(int j = 0; j < Input.size().width; j++){
                    returnTensor[{i, j}] = _Bias_[{1, j}] + Input[{i, j}];
                }
            }
            return returnTensor;
        }
};

//Convolutional Network
class Conv2D : public NetworkFunc{
    public:
    private:
        Tensor _Filter_;
};

class ReLU : public NetworkFunc{
    public:
        Tensor forward(Tensor& Input){
            Tensor newTensor({Input.size()});
            for(int i = 0; i < Input.size().height;i++){
                for(int j = 0; j < Input.size().width;j++){
                   if(Input[{i, j}] < 0){
                       newTensor[{i, j}] = 0;
                   } 
                   else{
                       newTensor[{i, j}] = Input[{i, j}];
                   }
                }
            }
            return newTensor;
        }
};