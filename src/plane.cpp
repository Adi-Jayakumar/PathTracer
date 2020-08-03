#include "plane.h"
#include <cmath>
#include <limits>

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

bool Plane::Intersect(Ray &ray, double &hit)
{
    double det = Vec::Dot(ray.d, n);

    // plane and ray parallel
    if (det == 0)
        return false;

    double t = Vec::Dot(n, p - ray.o) / det;
    if (t < PTUtility::EPSILON)
        return false;
    hit = t;
    return true;
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