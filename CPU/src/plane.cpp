#include "plane.h"


Plane::Plane(Vec _n, Vec _p)
{
    p = _p;
    n = _n;
}

bool Plane::Intersect(Ray &ray, double &hit)
{
    bool didHit = false;
    double t = std::numeric_limits<double>::max();
    double det = Vec::Dot(ray.d, n);

    // plane and ray not parallel
    if (fabs(det) >= PTUtility::EPSILON)
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

Vec Plane::Normal(Vec &)
{
    return n;
}

void Plane::Translate(Vec &x)
{
    p = p + x;
}