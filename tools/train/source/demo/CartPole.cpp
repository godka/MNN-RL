/*
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
*/
#include "CartPole.hpp"
#include <cmath>
#include <tuple>
#include <random>
#define PI 3.1415926535898

/*
Description:
    A pole is attached by an un-actuated joint to a cart, which moves along
    a frictionless track. The pendulum starts upright, and the goal is to
    prevent it from falling over by increasing and reducing the cart's
    velocity.

Source:
    This environment corresponds to the version of the cart-pole problem
    described by Barto, Sutton, and Anderson

Observation:
    Type: Box(4)
    Num     Observation               Min                     Max
    0       Cart Position             -4.8                    4.8
    1       Cart Velocity             -Inf                    Inf
    2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
    3       Pole Angular Velocity     -Inf                    Inf

Actions:
    Type: Discrete(2)
    Num   Action
    0     Push cart to the left
    1     Push cart to the right

    Note: The amount the velocity that is reduced or increased is not
    fixed; it depends on the angle the pole is pointing. This is because
    the center of gravity of the pole increases the amount of energy needed
    to move the cart underneath it

Reward:
    Reward is 1 for every step taken, including the termination step

Starting State:
    All observations are assigned a uniform random value in [-0.05..0.05]

Episode Termination:
    Pole Angle is more than 12 degrees.
    Cart Position is more than 2.4 (center of the cart reaches the edge of
    the display).
    Episode length is greater than 200.
    Solved Requirements:
    Considered solved when the average return is greater than or equal to
    195.0 over 100 consecutive trials.
*/

CartPole::CartPole() {
    this->gravity = 9.8;
    this->masscart = 1.0;
    this->masspole = 0.1;
    this->total_mass = (this->masspole + this->masscart);
    this->length = 0.5;  // actually half the pole's length
    this->polemass_length = (this->masspole * this->length);
    this->force_mag = 10.0;
    this->tau = 0.02;  // seconds between state updates

    // Angle at which to fail the episode
    this->theta_threshold_radians = 12 * 2 * PI / 360.;
    this->x_threshold = 2.4;

    this->seed();
    //this->state = nullptr;

    this->steps_beyond_done = -1;
}

int CartPole::seed() {
    return 42;
}

void CartPole::step(int action, std::vector<float>& obs, float& reward, bool& done) {
    auto x = this->state[0];
    auto x_dot = this->state[1];
    auto theta = this->state[2];
    auto theta_dot = this->state[3];
    auto force = 0.;

    if(action == 1){
        force = this->force_mag;
    }else{
        force = -this->force_mag;
    }
    auto costheta = std::cos(theta);
    auto sintheta = std::sin(theta);

    // For the interested reader:
    // https://coneural.org/florian/papers/05_cart_pole.pdf
    auto temp = (force + this->polemass_length * std::pow(theta_dot, 2) * sintheta) / this->total_mass;
    auto thetaacc = (this->gravity * sintheta - costheta * temp) / (this->length * (4.0 / 3.0 - this->masspole * std::pow(costheta,2) / this->total_mass));
    auto xacc = temp - this->polemass_length * thetaacc * costheta / this->total_mass;

    x = x + this->tau * x_dot;
    x_dot = x_dot + this->tau * xacc;
    theta = theta + this->tau * theta_dot;
    theta_dot = theta_dot + this->tau * thetaacc;

    this->state[0] = x;
    this->state[1] = x_dot;
    this->state[2] = theta;
    this->state[3] = theta_dot;

    done = x < -this->x_threshold || x > this->x_threshold || theta < -this->theta_threshold_radians || theta > this->theta_threshold_radians;

    if (!done){
        reward = 1.0;
    }else if (this->steps_beyond_done == -1){
        // Pole just fell!
        this->steps_beyond_done = 0;
        reward = 1.0;
    } else {
        this->steps_beyond_done += 1;
        reward = 0.0;
    }
    obs = this->state;
}

void CartPole::reset(std::vector<float>& obs) {
    std::vector<float> _state;
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(-0.05, 0.05);
    for (auto i = 0; i < 4; ++i){
        _state.push_back(dis(gen));
    }
    this->state = _state;
    this->steps_beyond_done = -1;
    obs = this->state;
}