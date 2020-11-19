#ifndef ValNetModels_hpp
#define ValNetModels_hpp

#include <MNN/expr/Module.hpp>
#include <MNN/expr/NN.hpp>

class MNN_PUBLIC ValNet : public MNN::Express::Module {
public:
    ValNet();

    virtual std::vector<MNN::Express::VARP> onForward(const std::vector<MNN::Express::VARP>& inputs) override;

    std::shared_ptr<MNN::Express::Module> fc1;
    std::shared_ptr<MNN::Express::Module> fc2;
    std::shared_ptr<MNN::Express::Module> fc3;
};

#endif
