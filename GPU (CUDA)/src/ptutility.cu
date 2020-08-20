#include <cmath>
#include <random>

#include "ptutility.h"
__device__ float PTUtility::Luma(const Vec &colour)
{
    return Vec::Dot(colour, Vec(0.2126, 0.7152, 0.0722));
}

__device__ float PTUtility::Random(curandState &state)
{
    return curand_uniform(&state);
}

__device__ Pair PTUtility::SolveQuadratic(float A, float B, float C)
{
    if (A < PTUtility::EPSILON)
        return {PTUtility::INF, PTUtility::INF};

    float discriminant = B * B - 4 * A * C;

    // Checking solutions exist
    if (discriminant < 0)
        return {PTUtility::INF, PTUtility::INF};

    // Finding both
    float t1 = (-B - sqrt(discriminant)) / (2 * A);
    float t2 = (-B + sqrt(discriminant)) / (2 * A);

    if (t2 <= t1)
    {
        float temp = t1;
        t1 = t2;
        t2 = temp;
    }

    return {t1, t2};
}

__device__ float Max(float a, float b)
{
    if (a > b)
        return a;
    else
        return b;
}