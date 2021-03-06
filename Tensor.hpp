/* Tensors are non-resizable arrays that can contain any value, nevertheless for this implementation, the default
value of float will be used. In addition, gradient descent algorithm will be implemented in this part of the 
code an hopefully, create a functional machine learning algorithm*/

#ifndef STD_IOSTREAM
#define STD_IOSTREAM
#include <iostream>
#endif

#ifndef STD_STDLIB
#define STD_STDLIB
#include <cstdlib>
#endif

#ifndef STD_TIME
#define STD_TIME
#include <ctime>
#endif 

#ifndef STD_VECTOR
#define STD_VECTOR
#include<vector>
#endif

//Abstraction Classes
struct TensorSize{
    /*
        A Structure representing the size or position of a tensor (matrix). The structure contains two variables
        which are height and width. 
    */
    long height;
    long width;
    long total(){
        return height * width;
    }
    
    TensorSize(const long& h, const long&w){
        height  = h;
        width   = w;
    }
    TensorSize(){
        height  = 0;
        width   = 0;
    }

    //operator overloads
    TensorSize& operator = (int* Initializer){
        this -> height  = Initializer[0];
        this -> width   = Initializer[1];
        return *this;
    }
    TensorSize& operator = (const TensorSize& Other){
        this -> height  = Other.height;
        this -> width   = Other.width;
        return *this;
    }
    bool operator == (const TensorSize &Other) const {
        if(this -> width != Other.width || this -> height != Other.height){
            return false;
        }
        return true;
    }
    bool operator != (const TensorSize &Other) const {
        if(this -> width != Other.width || this -> height != Other.height){
            return true;
        }
        return false;
    }
};

//MainClass Tensor to be used as the input matrix in to a neural network. The main class tensor provides 
//features 
class Tensor{   
    public:
        Tensor(const TensorSize& Size){
            _TensorContainer_   = new float*[Size.height];
            for(int i = 0; i < Size.height; i++){
                _TensorContainer_[i]    = new float[Size.width];
            }
            _Size_  = Size;
        }
        Tensor(const Tensor& Copy){
            _Size_              = Copy._Size_;
            _TensorContainer_   = new float*[_Size_.height];
            for(int i = 0; i < _Size_.height; i++){
                _TensorContainer_[i]    = new float[_Size_.width];
            }
            for(int i = 0; i < _Size_.height; i++){
                for(int j = 0; j < _Size_.width; j++){
                    _TensorContainer_[i][j] = Copy._TensorContainer_[i][j];
                }
            }
        }
        
        //Operator Overloads
        Tensor operator + (const Tensor& Other)const {
            if(Other._Size_ != this -> _Size_){
                throw "Unequal Tensor Size. Addition Failure";
            }
            for(long i = 0; i < Other._Size_.height; i++){
                for(long j = 0; j < Other._Size_.width; j++){
                    this -> _TensorContainer_[i][j] += Other._TensorContainer_[i][j];
                }
            }
            return *this;
        }
        Tensor operator - (const Tensor& Other)const {
            if(Other._Size_ != this -> _Size_){
                throw "Unequal Tensor Size. Addition Failure";
            }
            for(long i = 0; i < Other._Size_.height; i++){
                for(long j = 0; j < Other._Size_.width; j++){
                    this -> _TensorContainer_[i][j] -= Other._TensorContainer_[i][j];
                }
            }
            return *this;
        }
        Tensor operator * (const Tensor& Other)const {
            if(this -> _Size_.width != Other._Size_.height){
                throw "Cannot multiply matrix of conflicting size";
            }
            Tensor newTensor({this -> _Size_.height, Other._Size_.width});
            for(long i = 0;i < this -> _Size_.height; i++){
                for(long j = 0;j < Other._Size_.width; j++){
                    float value = 0;
                    for(long k = 0;k < this -> _Size_.width; k++){
                        value += this -> _TensorContainer_[i][k]*Other._TensorContainer_[k][j];
                    }
                    newTensor._TensorContainer_[i][j] = value;
                }
            }
            return newTensor;
        }
        Tensor operator * (const float& Other)const {
            for(long i = 0; i < _Size_.height; i++){
                for(long j = 0; j < _Size_.width; j++){
                    _TensorContainer_[i][j] *= Other;
                }
            }
            return *this;
        }
        Tensor& operator = (const Tensor& Other){
            //Delete the current stuff 
            if(this -> _Size_ != Other._Size_){
                for(int i = 0; i < this -> _Size_.height; i++){
                    delete[] this -> _TensorContainer_[i];
                }
                delete[] this -> _TensorContainer_;
                this -> _Size_  = Other._Size_;
                this -> _TensorContainer_ = new float*[this -> _Size_.height];
                for(int i = 0; i < this -> _Size_.height; i++){
                    this -> _TensorContainer_[i]   = new float[this -> _Size_.width];
                }
            }
            for(int i = 0; i < Other._Size_.height; i++){
                for(int j = 0; j < Other._Size_.width; j++){
                    this -> _TensorContainer_[i][j] = Other._TensorContainer_[i][j];
                }
            }
            return *this;
        }
        float& operator [] (const TensorSize& Position){
            return _TensorContainer_[Position.height][Position.width];
        }
        
        //Basic Functions
        TensorSize size() const{
            return _Size_;
        }
        float getValue(const TensorSize& Position) const{
            return _TensorContainer_[Position.height][Position.width];
        }
        void resize(const TensorSize& NewSize){
            for(long i = 0; i < _Size_.height; i++){
                delete[] _TensorContainer_[i];
            }
            delete[] _TensorContainer_;
            _TensorContainer_   = new float*[NewSize.height];
            for(int i = 0; i < NewSize.height; i++){
                _TensorContainer_[i]    = new float[NewSize.width];
            }
            _Size_  = NewSize;
        }
        /*Tensor* singularValueDecomposition(){
            Tensor S    = Eye({1, 1});
            Tensor V    = Eye({1, 1});
            Tensor D    = Eye({1, 1});
            return {S, V, D};
        }
        Tensor* luDecomposition(){
            Tensor L    = Eye({1, 1});
            Tensor U    = Eye({1, 1});
            return {L, U};
        }
        Tensor* diagonalize(){
            Tensor D    = Eye({1, 1});
            Tensor P    = Eye({1, 1});
            Tensor Pinv = Eye({1, 1});

            return {P, D, Pinv};
        }
        Tensor* qrDecomposition(){
            Tensor Q    = Eye({1, 1});
            Tensor R    = Eye({1, 1});
            return {Q, R};
        }
        Tensor* orthogonalDiagonalize(){
            Tensor P    = Eye({1, 1});
            Tensor D    = Eye({1, 1});

            return {P, D, P.rTranspose()};
        }*/

        //Matrix Properties
        Tensor eigenvalue();
        Tensor eigenvector();
        void transpose(){
            float **NewTensorContainer = new float* [_Size_.width];
            for(int i = 0; i < _Size_.width; i++){
                NewTensorContainer[i] = new float[_Size_.height];
                for(int j = 0; j < _Size_.height; j++){
                    NewTensorContainer[i][j] = _TensorContainer_[j][i];
                }
                delete[] _TensorContainer_[i];
            }
            delete[] _TensorContainer_;
            _TensorContainer_   = NewTensorContainer;
            _Size_              = {_Size_.width, _Size_.height};
        }
        Tensor rTranspose() const{
            Tensor newTensor({_Size_.width, _Size_.height});
            for(int i = 0; i < _Size_.width;i++){
                for(int j = 0; j < _Size_.height;j++){
                    newTensor[{i, j}] = _TensorContainer_[j][i];
                }
            }
            return newTensor;
        }
        Tensor rank();
        Tensor null();

        //Other Matrix stuffs
        bool symmetric();

        //Null Rank and stuff
        //Gradient Tracker
        /* Create a tensor with value difference of 0.01 for gradient calculation purposes */
        Tensor gradTensor(const float& diff) const{
            Tensor newTensor({_Size_});
            for(int i = 0;i < _Size_.height; i++){
                for(int j = 0; j < _Size_.width; j++){
                    newTensor._TensorContainer_[i][j] = _TensorContainer_[i][j] - diff;
                }
            }
            return newTensor;
        }

        /*Calculate the Average of a Tensor for Gradient Descent Calculation*/
        Tensor avgTensor() const{
            Tensor average({_Size_.width, 1});
            for(int i = 0; i < _Size_.width; i++){
                float sum = 0;
                for(int j = 0; j < _Size_.height; j++){
                    sum += _TensorContainer_[j][i] / _Size_.height;
                }
                average._TensorContainer_[i][0] = sum;
            }
            return average;
        }

        //Destructor
        ~Tensor(){
            for(long i = 0; i < _Size_.height; i++){
                delete[] _TensorContainer_[i];
            }
            delete[] _TensorContainer_;
        }
    protected:
        float** _TensorContainer_;
        TensorSize _Size_;
        
};

//Tensor Initializer Class
class Eye : public Tensor{
    public:
        Eye(const TensorSize &Size) : Tensor(Size){
            if(Size.height != Size.width){
                throw "Identity matrix are square matrices";
            }
            for(int i = 0; i < Size.height; i++){
                for(int j = 0; j < Size.width; j++){
                    _TensorContainer_[i][j] = 0;
                }
            }
            for(int i = 0; i < Size.height; i++){
                _TensorContainer_[i][i] = 1;
            }
        }
};
class Ones : public Tensor{
    public: 
        Ones(const TensorSize &Size) : Tensor(Size){
            for(long i = 0; i < Size.height; i++){
                for(long j = 0; j <  Size.width; j++){
                    _TensorContainer_[i][j] = 1;
                }
            }
        }
};
class Zeros : public Tensor{
    public: 
        Zeros(const TensorSize &Size) : Tensor(Size){
            for(int i = 0; i < Size.height; i++){
                for(int j = 0; j < Size.width; j++){
                    _TensorContainer_[i][j] = 0;
                }
            }
        }
};
class Random : public Tensor{
    public: 
        Random(const TensorSize& Size) : Tensor(Size){
            for(long i = 0; i < Size.height; i++){
                for(long j = 0; j < Size.width; j++){
                    srand(time(0));
                    _TensorContainer_[i][j] = rand();
                }
            }
        }
};
class Fill : public Tensor{
    public: 
        Fill(const TensorSize& Size,const float &Value) : Tensor(Size){
            for(long i = 0; i < Size.height; i++){
                for(long j = 0; j < Size.width; j++){
                    _TensorContainer_[i][j] = Value;
                }
            }
        }
};

//Tensor Functions, Functions Involving Tensors either to print or other stuff
void print(const Tensor& Other){
    for(long i = 0; i < Other.size().height; i++){
        for(long j = 0; j < Other.size().width; j++){
            std::cout<<Other.getValue({i, j})<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;   
}
void print(const TensorSize& Other){
    std::cout<<"Height   : "<<Other.height<<std::endl;
    std::cout<<"Width    : "<<Other.width<<std::endl;
}
void print(const float& Other){
    std::cout<<Other<<std::endl;
}
Tensor max(const Tensor& Input){
    Tensor newTensor({Input.size().height, 2});
    for(long i = 0; i < Input.size().height; i++){
        float maxMarker = Input.getValue({i, 0});
        long maxIndex = 0;
        for(long j = 1; j < Input.size().width; j++){
            if(maxMarker < Input.getValue({i, j})){
                maxMarker = Input.getValue({i, j});
                maxIndex  = j;
            }
        }
        newTensor[{i, 0}] = maxMarker;
        newTensor[{i, 1}] = maxIndex;
    }
    return newTensor;
}
Tensor flatten(const std::vector <std::vector<Tensor> >& Input){
    Tensor newTensor({Input.size(), Input[0].size() * Input[0][0].size().total()});
    long count = 0;
    for(int i = 0; i < Input.size(); i++){
        for(auto j = Input[i].begin(); j < Input[i].end(); i++){
            for(long k = 0; k < j -> size().height; ++k){
                for(long l = 0; l < j -> size().width; ++l){
                    newTensor[{i, count}]  = j -> getValue({k, l});
                    ++count;
                }
            }
        }
    }
    return newTensor;
}

