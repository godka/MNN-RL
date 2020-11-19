#include "Val.hpp"

using namespace MNN;
using namespace MNN::Express;

ValNet::ValNet()
{
    this->fc1.reset(NN::Linear(4, 100));
    this->fc2.reset(NN::Linear(100, 100));
    this->fc3.reset(NN::Linear(100, 1));
    registerModel({this->fc1, this->fc2, this->fc3});
}

std::vector<MNN::Express::VARP> ValNet::onForward(const std::vector<MNN::Express::VARP>& inputs)
{
    using namespace Express;
    VARP x = inputs[0];
    x      = fc1->forward(x);
    x      = _Relu(x);
    x      = fc2->forward(x);
    x      = _Relu(x);
    x      = fc3->forward(x);
    return {x};
}
