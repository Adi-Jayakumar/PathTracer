#include <cmath>
#include <random>

#include "ptutility.h"
__device__ double PTUtility::Luma(const Vec &colour)
{
    return Vec::Dot(colour, Vec(0.2126, 0.7152, 0.0722));
}

__device__ double PTUtility::Random(curandState &state)
{
    return curand_uniform(&state);
}



__device__ Pair PTUtility::SolveQuadratic(double A, double B, double C)
{
    if(A < PTUtility::EPSILON)
        return {PTUtility::INF, PTUtility::INF};
    
    double discriminant = B * B - 4 * A * C;

    // Checking solutions exist
    if (discriminant < 0)
        return { PTUtility::INF, PTUtility::INF };

    // Finding both
    double t1 = (-B - sqrt(discriminant)) / (2 * A);
    double t2 = (-B + sqrt(discriminant)) / (2 * A);

    if (t2 <= t1)
    {   
        double temp = t1;
        t1 = t2;
        t2 = temp;
    }

    return { t1, t2 };
}


__device__ double Max(double a, double b)
{
    if(a > b)
        return a;
    else
        return b;
}