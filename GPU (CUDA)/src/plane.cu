#include "plane.cuh"

__device__ Plane::Plane(Vec _n, Vec _p)
{
    p = _p;
    n = _n;
}

__device__ bool Plane::Intersect(Ray &ray, double &hit)
{
    bool didHit = false;
    double t = PTUtility::INF;
    double det = Vec::Dot(ray.d, n);

    // plane and ray not parallel
    if (fabs(det) != 0)
    {
        t = Vec::Dot(n, p - ray.o) / det;
        if (t > PTUtility::EPSILON)
        {
            hit = t;
            didHit = true;
        }
    }
    return didHit;
}

__device__ Vec Plane::Normal(Vec &x)
{
    return n;
}

__device__ void Plane::Translate(Vec &x)
{
    p = p + x;
}