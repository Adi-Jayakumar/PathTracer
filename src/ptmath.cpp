#include <cmath>
#include <random>
#include "ptmath.h"

double PTMath::Luma(const Vec &colour)
{
    return Vec::Dot(colour, Vec(0.2126, 0.7152, 0.0722));
}

double PTMath::Random()
{
    static thread_local std::mt19937 prng(std::random_device{}());
    static thread_local std::uniform_real_distribution<double> dist(0, 1.0);
    return dist(prng);
}
