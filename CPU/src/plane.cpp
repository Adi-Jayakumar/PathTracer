#include "plane.h"
#include <cmath>
#include <limits>

Plane::Plane(Vec _n, Vec _p)
{
    p = _p;
    n = _n;
}

bool Plane::Intersect(Ray &ray, double &hit, std::shared_ptr<std::pair<double, double>> values)
{
    bool didHit = false;
    double t = std::numeric_limits<double>::max();
    double det = Vec::Dot(ray.d, n);

    // plane and ray not parallel
    if (det != 0)
    {
        t = Vec::Dot(n, p - ray.o) / det;
        if (t > PTUtility::EPSILON)
        {
            hit = t;
            didHit = true;
        }
    }

    if (values != nullptr)
    {
        values->first = t;
        values->second = t;
    }
    return didHit;
}

Vec Plane::Normal(Vec &x)
{
    return n;
}

void Plane::Translate(Vec &x)
{
    p = p + x;
}

bool Plane::IsOnSkin(Vec &x)
{
    return fabs(Vec::Dot(n, x - p)) < PTUtility::EPSILON;
}
