#include "NeuralLib.hpp"

//Example Code here
class LinearClassifier : public nl::Model{
    public:
        LinearClassifier(const long& InputSize, const long&OutputSize, const long& NetworkDepth, const long& NetworkSize){
            _Sequence_.push_back(new nl::Linear(InputSize, NetworkSize, true));
            for(long i = 1; i < NetworkDepth - 1; i++){
                _Sequence_.push_back(new nl::Linear(NetworkSize, NetworkSize, true));
            }
            _Sequence_.push_back(new nl::Linear(NetworkSize, OutputSize, true));
        }
        nl::Tensor forward(const nl::Tensor& Input){
            nl::Tensor Output = _Sequence_[0] -> forward(Input);
            for(long i = 1; i < _Sequence_.size(); i++){
                Output  = _Sequence_[i] -> forward(Output);
            }
            return nl::max(Output);
        }
        ~LinearClassifier(){
            for(auto i = _Sequence_.begin(); i < _Sequence_.end(); i++){
                delete *i;
            }
        }
    private:
        std::vector <nl::NetworkFunc*> _Sequence_;
};

int main(){
    nl::Tensor newTensor    = nl::Zeros({64, 3});
    LinearClassifier model(3, 10, 256, 20);
    for(long i = 0; i < 1000; i++){
        model.forward(newTensor);
        std::cout<<i<<std::endl;
    }
    nl::Tensor Output = model.forward(newTensor);
    print(Output);
}