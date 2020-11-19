#ifndef PolicyNetModels_hpp
#define PolicyNetModels_hpp

#include <MNN/expr/Module.hpp>
#include <MNN/expr/NN.hpp>

class MNN_PUBLIC PolicyNet : public MNN::Express::Module {
public:
    PolicyNet();

    virtual std::vector<MNN::Express::VARP> onForward(const std::vector<MNN::Express::VARP>& inputs) override;

    std::shared_ptr<MNN::Express::Module> fc1;
    std::shared_ptr<MNN::Express::Module> fc2;
    std::shared_ptr<MNN::Express::Module> fc3;
};

#endif
