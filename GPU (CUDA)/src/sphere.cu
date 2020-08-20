#include "pair.h"
#include "sphere.h"
#include <cmath>
#include <limits>

__device__ Sphere::Sphere(float _r, Vec _c)
{

    r = _r;
    c = _c;
}

__device__ bool Sphere::Intersect(Ray &ray, float &hit)
{
    float A = Vec::Dot(ray.d, ray.d);
    float B = 2 * Vec::Dot(ray.o, ray.d) - 2 * Vec::Dot(c, ray.d);
    float C = Vec::Dot(ray.o, ray.o) + Vec::Dot(c, c) - 2 * Vec::Dot(ray.o, c) - r * r;
    bool didHit = false;

    Pair vals = PTUtility::SolveQuadratic(A, B, C);
    hit = PTUtility::INF;
    if (vals.first > PTUtility::EPSILON && vals.first != PTUtility::INF)
    {
        didHit = true;
        hit = vals.first;
    }
    else if (vals.second > PTUtility::EPSILON && vals.second != PTUtility::INF)
    {
        didHit = true;
        hit = vals.second;
    }
    return didHit;
};

__device__ Vec Sphere::Normal(Vec &v)
{
    return (v - c) / r;
}

__device__ void Sphere::Translate(Vec &x)
{
    c = c + x;
}