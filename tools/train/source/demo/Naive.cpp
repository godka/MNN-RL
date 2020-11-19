#include "Naive.hpp"
#include <cmath>
#include <tuple>
#include <random>

Naive::Naive() {
    this->steps = 0;
    this->seed();
}

int Naive::seed() {
    return 42;
}

void Naive::step(int action, std::vector<float>& obs, float& reward, bool& done) {
    
    reward = (float)action;
    this->state[this->steps] = (float)action;
    this->steps++;
    if (this->steps >= 4)
    {
        done = true;
    }else{
        done = false;
    }
    obs = this->state;
}

void Naive::reset(std::vector<float>& obs) {
    std::vector<float> _state(4);
    this->steps = 0;
    this->state = _state;
    obs = this->state;
}