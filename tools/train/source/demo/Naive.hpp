#ifndef Naive_hpp
#define Naive_hpp
#include <vector>
class Naive {
public:
    Naive();
    int seed();
    void step(int action, std::vector<float>& obs, float& reward, bool& done);
    void reset(std::vector<float>& obs);

private:
    std::vector<float> state;
    int steps_beyond_done;
    int steps;
};
#endif
