#ifndef A2C_hpp
#define A2C_hpp
#include <vector>
#include <memory>
#include <string>

#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/NN.hpp>
#include "Policy.hpp"
#include "Val.hpp"
#include "ADAM.hpp"

class A2C {
public:
    A2C(int s_info, int a_dim, double learning_rate);
    void Train(std::vector<std::vector<float>>& s_batch,
                std::vector<int32_t>& a_batch, 
                std::vector<float>& r_batch);
    std::vector<float> Predict(std::vector<float>& obs);
    // void Load(std::string& filename);
    // void Save(std::string& filename);
    
private:
    std::shared_ptr<MNN::Express::Module> policy_;
    std::shared_ptr<MNN::Express::Module> val_;
    int s_info;
    int a_dim;

    std::shared_ptr<MNN::Train::ADAM> policy_adam_;
    std::shared_ptr<MNN::Train::ADAM> val_adam_;

    std::shared_ptr<MNN::Express::Executor> exe;
    MNN::Express::VARP Policy_Loss(MNN::Express::VARP pi, MNN::Express::VARP oneHotActions, MNN::Express::VARP reward);
    MNN::Express::VARP Val_Loss(MNN::Express::VARP val, MNN::Express::VARP reward);
    std::vector<float> ComputeR(std::vector<float>& r_batch);
protected:

};
#endif
