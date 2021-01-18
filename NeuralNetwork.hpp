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
        virtual Tensor forward(const Tensor& Input) = 0;
};

//Template for all class (You can Declare a Sequential with this interface with a vector)
class NetworkFunc{
    public: 
        virtual Tensor forward(const Tensor& Input) = 0;
};

//Standard NN Network 
class Linear : public NetworkFunc{
    public:
        Linear(const long& Input, const long& Output, const bool& Bias) : _Weight_(Fill({Input, Output}, 0.01)), _Bias_(Fill({Output, 1}, 0.01)){
            //Weigth initialized of size m x n : Input x Output. If there are 7 layers of Input, then the
            //Resulting Matrix will be 7 * Output
            _UseBias_   = Bias;
        }
        Tensor forward(const Tensor& Input){
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
        Tensor _AddBias_(const Tensor& Input){
            Tensor returnTensor({Input.size()});
            for(int i = 0; i < Input.size().height;i++){
                for(int j = 0; j < Input.size().width; j++){
                    returnTensor[{i, j}] = _Bias_[{1, j}] + Input.getValue({i, j});
                }
            }
            return returnTensor;
        }
};

//Convolutional Network
class Conv2D : public NetworkFunc{
    public:
        Conv2D(const long& InputChannel, const long& OutputChannel, const long& Stride, const long& KernelSize){
            _InputSize_     = InputChannel;
            _OutputSize_    = OutputChannel;
            _Stride_        = Stride;
            
            //Initialize Kernel
            for(int i = 0;i < OutputChannel; i++){
                _Kernel_.push_back(Random({KernelSize, KernelSize}));
            }
        }

        std::vector<Tensor> forward(const std::vector<Tensor> &Input){
            //slide kernel over tensor
            //Check for incompatibility of the tensor
            if(Input.size() != _InputSize_){
                throw "The input tensor does not match the neural network requirement";
            }

            /*find the kernel stuff over here*/
            TensorSize Size         = Input[0].size();
            long Kernel             = _Kernel_[0].size().width;
            TensorSize outputSize   = {(Size.height - Kernel) / _Stride_, (Size.width - Kernel) / _Stride_};
            if((Size.height - Kernel) / _Stride_ != outputSize.height || (Size.width - Kernel) / _Stride_ != outputSize.width){
                throw "Stride cannot match tensor and thereofore cannot give a good output";
            }
            std::vector<Tensor> Output;

            for(auto ptr = _Kernel_.begin();ptr != _Kernel_.end(); ptr++){
                Tensor Channel({outputSize});
                for(long i = 0; i < Size.height - Kernel; i+=_Stride_){
                    for(long j = 0; j < Size.width - Kernel; j+=_Stride_){
                        float sum = 0;
                        for(long k = 0; k < Kernel; k++){
                            for(long l = 0; l < Kernel; l++){
                                TensorSize Position = {i + k, j + l};
                                for(auto ptr2 = Input.begin(); ptr2 < Input.end(); ptr2++){
                                    sum += ptr2 -> getValue(Position) * (*ptr)[{k, l}];
                                }
                            }
                        }
                        Output.push_back(Channel);
                    }
                }
            }
            return Output;
        }
    private:
        std::vector <Tensor> _Kernel_;
        long _Stride_;       
        long _InputSize_;
        long _OutputSize_;
};

//Activation Functions
//Create Base Class for Activation Functions
class ActivationFunction : public NetworkFunc{
    public:
        Tensor forward(const Tensor& Input){
            Tensor newTensor(Input.size());
            for(long i = 0; i < Input.size().height; i++){
                for(long j = 0; j < Input.size().width; j++){
                    newTensor[{i, j}] = func(Input.getValue({i, j}));
                }
            }
            return newTensor;
        } 
        virtual float func(const float& Input) = 0;
};
class ReLU : public ActivationFunction{
    public:
        float func(const float& Input){
            if(Input > 0){
                return Input;
            }
            else{
                return 0;
            }
        }
};
class Sigmoid : public ActivationFunction{
    public:
        float func(const float& Input){
            return 1 / (1 + exp(-Input));
        }
};
class Tanh : public ActivationFunction{
    public:
        float func(const float& Input){
            return tanh(Input);
        }
};