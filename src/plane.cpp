#include <limits>
#include "plane.h"

Plane::Plane(Vec _n, Vec _p)
{
    p = _p;
    n = _n;
}

Plane::Plane(const Plane &plane)
{
    p = plane.p;
    n = plane.n;
}

double Plane::Intersect(Ray &ray)
{
    double det = Vec::Dot(ray.d, n);
    if (det == 0)
        return std::numeric_limits<double>::max();

    double t = Vec::Dot(n, p - ray.o) / det;
    if (t < PTUtility::EPSILON)
        return std::numeric_limits<double>::max();
    return t;
}

Vec Plane::Normal(Vec &x)
{
    return n;
}

void Plane::Translate(Vec &x)
{
    p = p + x;
}