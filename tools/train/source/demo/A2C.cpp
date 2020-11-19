#include <MNN/expr/Executor.hpp>
#include <memory>
#include <cmath>
#include <iostream>
#include <vector>
#include <MNN/expr/NN.hpp>
#include "ADAM.hpp"
#include "Loss.hpp"
#include "OpGrad.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include "A2C.hpp"
using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

#define GAMMA 0.99

MNN::Express::VARP A2C::Policy_Loss(Express::VARP pi, Express::VARP oneHotActions, Express::VARP adv) {
    MNN_ASSERT(pi->getInfo()->dim == oneHotActions->getInfo()->dim);
    MNN_ASSERT(oneHotActions->getInfo()->dim == reward->getInfo()->dim);
    auto pi_loss = _Negative(
        _ReduceMean(
            _ReduceSum(
                _Log(pi) * oneHotActions, {1}
                ) * adv, {}
            )
        );
    return pi_loss;
}

MNN::Express::VARP A2C::Val_Loss(Express::VARP val, Express::VARP reward) {
    auto val_loss = _MSE(val, reward);
    return val_loss;
}

// void A2C::Load(std::string& filename) {
//     // Load snapshot
//     auto para = Variable::load(filename.c_str());
//     this->model->loadParameters(para);
// }

// void A2C::Save(std::string& filename) {
//     return;
// }

A2C::A2C(int s_info, int a_dim, double learning_rate) {
    this->exe = Executor::getGlobalExecutor();
    BackendConfig config;
    this->exe->setGlobalExecutorConfig(MNN_FORWARD_AUTO, config, 4);
    std::shared_ptr<Module> _policy(new PolicyNet());
    this->policy_ = std::move(_policy);
    std::shared_ptr<Module> _val(new ValNet());
    this->val_ = std::move(_val);

    std::shared_ptr<ADAM> p_adam(new ADAM(this->policy_));
    p_adam->setLearningRate(1e-4);
    this->policy_adam_ = std::move(p_adam);

    std::shared_ptr<ADAM> v_adam(new ADAM(this->val_));
    v_adam->setLearningRate(1e-4);
    this->val_adam_ = std::move(v_adam);

    this->policy_->clearCache();
    this->val_->clearCache();

    this->exe->gc(Executor::FULL);
    this->exe->resetProfile();

    this->policy_->setIsTraining(true);
    this->val_->setIsTraining(true);

    this->s_info = s_info;
    this->a_dim = a_dim;
    std::cout << "A2C::Init Done" << std::endl;
}

std::vector<float> A2C::Predict(std::vector<float>& obs)
{
    auto states = _Input({1, this->s_info}, NHWC, halide_type_of<float>());
    ::memcpy(states->writeMap<float>(), obs.data(), obs.size() * sizeof(float));
    auto pi = this->policy_->forward(states);
    auto pi_ptr = pi->readMap<float>();

    std::vector<float> pi_ret(this->a_dim);
    for(auto i = 0; i < this->a_dim; ++i){
        pi_ret[i] = pi_ptr[i];
    }

    return pi_ret;
}

void A2C::Train(std::vector<std::vector<float>>& s_batch,
                std::vector<int32_t>& a_batch, 
                std::vector<float>& r_batch) {
    // from vector to VARP
    auto batch_size = s_batch.size();
    // generate tmp state
    std::vector<float> s_tmp;
    for (auto &t: s_batch){
        for (auto &l: t){
            s_tmp.push_back(l);
        }
    }
    // compute real reward
    auto R_batch = this->ComputeR(r_batch);

    auto states = _Input({int(batch_size), this->s_info}, NHWC, halide_type_of<float>());
    auto actions = _Input({int(batch_size), 1}, NHWC, halide_type_of<int32_t>());
    auto rewards = _Input({int(batch_size), 1}, NHWC, halide_type_of<float>());
    auto advs = _Input({int(batch_size), 1}, NHWC, halide_type_of<float>());

    ::memcpy(states->writeMap<float>(), s_tmp.data(), s_tmp.size() * sizeof(float));
    ::memcpy(actions->writeMap<int32_t>(), a_batch.data(), a_batch.size() * sizeof(int32_t));
    ::memcpy(rewards->writeMap<float>(), R_batch.data(), R_batch.size() * sizeof(float));

    auto actionOneHot = _OneHot(_Cast<int32_t>(actions), _Scalar<int>(this->a_dim), _Scalar<float>(1.0f),
                                _Scalar<float>(0.0f));
    
    auto pi = this->policy_->forward(states);
    auto val = this->val_->forward(states);
    auto val_mem = val->readMap<float>();
    std::vector<float> adv(batch_size);
    for (auto i = 0; i < batch_size; ++i) {
        adv[i] = R_batch[i] - val_mem[i];
    }
    ::memcpy(advs->writeMap<float>(), adv.data(), adv.size() * sizeof(float));

    auto p_loss = this->Policy_Loss(pi, actionOneHot, advs);
    this->policy_adam_->step(p_loss);

    auto v_loss = this->Val_Loss(val, rewards);
    this->val_adam_->step(v_loss);
}

std::vector<float> A2C::ComputeR(std::vector<float>& r_batch)
{
    auto batch_size = r_batch.size();
    std::vector<float> R_batch(batch_size);
    for(auto i = 0; i < batch_size - 1; ++i)
    {
        auto t = batch_size - 1 - i;
        R_batch[t] = r_batch[t] + GAMMA * R_batch[t + 1];
    }
    return R_batch;
}

