#include "A2C.hpp"
//#include "CartPole.hpp"
#include "Naive.hpp"
#include <memory>
#include <vector>
#include <iostream>
#include <math.h>
#include <random>
#include <limits>
#include "DemoUnit.hpp"

#define A_DIM 2

class ReinforcementLearning : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 1) {
            std::cout << "usage: ./runTrainDemo.out ReinforcementLearning" << std::endl;
            return 0;
        }
        std::shared_ptr<A2C> a2c(new A2C(4, 2, 1e-4));
        std::shared_ptr<Naive> env(new Naive());

        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(0., 1.);

        std::vector<float_t> obs;
        float reward;
        bool done;

        for (auto t = 0; t < 10000; ++t) {
            std::vector<std::vector<float_t>> s_batch;
            std::vector<int> a_batch;
            std::vector<float_t> r_batch;

            env->reset(obs);
            for (auto step = 0; step < 500; ++step) {
                s_batch.push_back(obs);

                auto prob = a2c->Predict(obs);

                // gumble samling
                std::vector<float_t> gumble(A_DIM);
                double tmp = -std::numeric_limits<double>::infinity();
                int tmp_idx = 0;
                for (int n = 0; n < A_DIM; ++n) {
                    gumble[n] = -std::log(-std::log(dis(gen))) + std::log(prob[n]);
                    if (gumble[n] > tmp){
                        tmp_idx = n;
                        tmp = gumble[n];
                    }
                }
                auto action = tmp_idx;

                env->step(action, obs, reward, done);

                a_batch.push_back(action);
                r_batch.push_back(reward);

                if (done){
                    auto cum = 0.;
                    for (auto &t: r_batch){
                        cum += t;
                    }
                    std::cout << cum << std::endl;
                    break;
                }
            }
            a2c->Train(s_batch, a_batch, r_batch);
            
        }
        return 0;   
    }
};
DemoUnitSetRegister(ReinforcementLearning, "ReinforcementLearning");
