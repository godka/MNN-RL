#ifndef CartPole_hpp
#define CartPole_hpp
#include <vector>
class CartPole {
public:
    CartPole();
    int seed();
    void step(int action, std::vector<float>& obs, float& reward, bool& done);
    void reset(std::vector<float>& obs);

private:
    float gravity;
    float masscart;
    float masspole;
    float total_mass;
    float length;  // actually half the pole's length
    float polemass_length;
    float force_mag;
    float tau;  // seconds between state updates

    // Angle at which to fail the episode
    float theta_threshold_radians;
    float x_threshold;

    std::vector<float> state;
    int steps_beyond_done;
};
#endif
