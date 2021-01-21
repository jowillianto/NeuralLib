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
class LinearFunc{
    public: 
        virtual Tensor forward(const Tensor& Input) = 0;
};
class ConvFunc{
    public: 
        virtual std::vector<std::vector<Tensor> > forward(const std::vector<std::vector<Tensor> >& Input) = 0;
};
//Standard NN Network 
class Linear : public LinearFunc{
    public:
        Linear(const long& Input, const long& Output, const bool& Bias = true) : _Weight_(Random({Input, Output})), _Bias_(Random({Input, Output})){
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
class Conv1D : public ConvFunc{};
class Conv2D : public ConvFunc{
    public:
        Conv2D(const long& InputChannel, const long& OutputChannel, const long& KernelSize, const long& Stride){
            _InputChannel_  = InputChannel;
            _OutputChannel_ = OutputChannel;
            _KernelSize_    = KernelSize;
            _Stride_        = Stride;
            for(long i = 0; i < _OutputChannel_; i++){
                _Kernel_.push_back(Ones({_KernelSize_, _KernelSize_}));
            }
        }
        std::vector<std::vector<Tensor> > forward(const std::vector<std::vector<Tensor> >& Input){
            std::vector<std::vector<Tensor> > Output; 
            //main Check to throw Erros
            if(Input[0].size() != _InputChannel_){
                throw "Incompatible Input Channel Size";
            }
            TensorSize inputSize    = Input[0][0].size();
            TensorSize outputSize   = {(inputSize.height - _KernelSize_) / _Stride_, (inputSize.width - _KernelSize_) / _Stride_};
            for(auto i = Input.begin(); i != Input.end(); i++){
                std::vector<Tensor> pushChannel;
                for(auto j = _Kernel_.begin(); j != _Kernel_.end(); j++){
                    Tensor pushTensor({outputSize});
                    long x = 0, y = 0;
                    for(long k = 0; k < inputSize.height; k+=_Stride_){
                        for(long l = 0; l < inputSize.width; l+= _Stride_){
                            float sum = 0;
                            for(long m = 0; m < _KernelSize_; m++){
                                for(long n = 0; n < _KernelSize_; n++){
                                    TensorSize position = {k + l, m + n};
                                    for(auto o = i -> begin(); o != i -> end(); o++){
                                        sum += o -> getValue(position) * j -> getValue({m, n});
                                    }
                                }
                            }
                            pushTensor[{x, y}] = sum;
                            x+=1; y+=1;
                        }
                    }
                    pushChannel.push_back(pushTensor);
                }
                Output.push_back(pushChannel);
            }
            return Output;
        }
    private: 
        std::vector <Tensor> _Kernel_;
        long _InputChannel_;
        long _OutputChannel_;
        long _KernelSize_;
        long _Stride_;
};
class Conv3D : public ConvFunc{};

//Pooling Layers
class MaxPool1D : public ConvFunc{};
class MaxPool2D : public ConvFunc{};
class MaxPool3D : public ConvFunc{};
class MinPool1D : public ConvFunc{};
class MinPool2D : public ConvFunc{};
class MinPool3D : public ConvFunc{};
class AvgPool1D : public ConvFunc{};
class AvgPool2D : public ConvFunc{};
class AvgPool3D : public ConvFunc{};


//Activation Functions
//Create Base Class for Activation Functions. Never use the class ActivationFunction, use network func
class ActivationFunction : public LinearFunc, public ConvFunc{
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
        std::vector<std::vector<Tensor> > forward(const std::vector<std::vector<Tensor> >& Input){
            std::vector<std::vector<Tensor> > Output;
            for(long i = 0; i < Input.size(); i++){
                std::vector<Tensor> pushChannel;
                for(long j = 0; j < Input[i].size(); j++){
                    Tensor pushTensor({Input[i][j].size()});
                    for(long k = 0; k < Input[i][j].size().height; k++){
                        for(long l = 0; l < Input[i][j].size().width; l++){
                            pushTensor[{k, l}] = func(Input[i][j].getValue({k, l}));
                        }
                    }
                    pushChannel.push_back(pushTensor);
                }
                Output.push_back(pushChannel);
            }
            return Output;
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
class LeakyRelu : public ActivationFunction{
    public:
        float func(const float& Input){
            if(Input > 0){
                return Input;
            }
            else{
                return _Gradient_ * Input;
            }
        }
    private:
        float _Gradient_;
};
class HardTanh : public ActivationFunction{
    public:
        float func(const float& Input){
            if(Input > 1){
                return 1;
            }
            else if(Input < -1){
                return -1;
            }
            else{
                return Input;
            }
        }
};
class HardSigmoid : public ActivationFunction{
    public: 
        float func(const float& Input){
            if(Input > 3){
                return 1;
            }
            else if(Input < -3){
                return 0;
            }
            else{
                return Input / 6 + 0.5;
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
class Softmax : public LinearFunc{
    public:
        Tensor forward(const Tensor& Input){
            Tensor newTensor(Input.size());
            //Calculate Softmax Sum
            for(long i = 0; i < Input.size().height; i++){
                float sum = 0;
                for(long j = 0; j < Input.size().width; j++){
                    sum += exp(Input.getValue({i, j}));
                }
                for(long j = 0; j < Input.size().width; j++){
                    newTensor[{i, j}] = exp(Input.getValue({i, j})) / sum;
                }
            }
            return newTensor;
        }
};
class LogSoftMax : public LinearFunc{
    public:
        Tensor forward(const Tensor& Input){
            Tensor newTensor(Input.size());
            //Calculate Softmax Sum
            for(long i = 0; i < Input.size().height; i++){
                float sum = 0;
                for(long j = 0; j < Input.size().width; j++){
                    sum += exp(Input.getValue({i, j}));
                }
                for(long j = 0; j < Input.size().width; j++){
                    newTensor[{i, j}] = log(exp(Input.getValue({i, j})) / sum);
                }
            }
            return newTensor;
        }
};
