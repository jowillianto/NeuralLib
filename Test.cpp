#include "NeuralLib.hpp"

/*Example of Linear Neural Network with Unknown Model*/
class LinearClassifier : public nl::Model{
    public:
        LinearClassifier(const int& InputSize, const int& OutputSize, const int& NetworkWidth, const int& NetworkDepth){
            _Sequence_.push_back(new nl::Linear(InputSize, NetworkWidth, true));
            _Sequence_.push_back(new nl::Sigmoid);
            for(int i = 1; i < NetworkDepth - 1; i++){
                _Sequence_.push_back(new nl::Linear(NetworkDepth, NetworkDepth, true));
                _Sequence_.push_back(new nl::Sigmoid);
            }
            _Sequence_.push_back(new nl::Linear(NetworkDepth, OutputSize, true));
        }
        nl::Tensor forward(const nl::Tensor& Input){
            nl::Tensor Output = _Sequence_[0] -> forward(Input);
            nl::print(Output);
            std::cout<<std::endl;
            for(int i = 1; i < _Sequence_.size(); i++){
                Output = _Sequence_[i] -> forward(Output);
                nl::print(Output);
                std::cout<<std::endl;
            }
            return Output;
        }
        ~LinearClassifier(){
            for(auto i = _Sequence_.begin(); i < _Sequence_.end(); i++){
                delete (*i);
            }
        }
    private:
        std::vector <nl::LinearFunc*> _Sequence_;
};

int main(){
    LinearClassifier func(10, 2, 20, 20);
    nl::Tensor Input    = nl::Zeros({1, 10});
    func.forward(Input);
    std::cin.get();
}
