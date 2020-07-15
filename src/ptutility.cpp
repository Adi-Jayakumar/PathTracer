#include <cmath>
#include <random>
#include "ptutility.h"

double PTUtility::Luma(const Vec &colour)
{
    return Vec::Dot(colour, Vec(0.2126, 0.7152, 0.0722));
}

double PTUtility::Random()
{
    static thread_local std::mt19937 prng(std::random_device{}());
    static thread_local std::uniform_real_distribution<double> dist(0, 1.0);
    return dist(prng);
}

double PTUtility::SolveQuadratic(double A, double B, double C)
{
    double discriminant = B * B - 4 * A * C;

    // Checking solutions exist
    if (discriminant < 0)
        return std::numeric_limits<double>::max();

    // Finding both
    double t1 = (-B - sqrt(discriminant)) / (2 * A);
    double t2 = (-B + sqrt(discriminant)) / (2 * A);

    if (t2 <= t1)
        std::swap(t1, t2);

    // We now know that t1 < t2
    if (t2 < 0)
        return std::numeric_limits<double>::max();
    else if (t1 >= 0)
    {
        if (t1 > PTUtility::EPSILON)
            return t1;
        else
            return std::numeric_limits<double>::max();
    }
    else if (t1 < 0)
    {
        if (t2 > PTUtility::EPSILON)
            return t2;
        else
            return std::numeric_limits<double>::max();
    }

    return std::numeric_limits<double>::max();
}
