#include <limits>
#include "plane.h"

Plane::Plane(Vec _n, Vec _p, Vec _e, Vec _c, Surface _s)
{
    p = _p;
    e = _e;
    c = _c;

    n = _n;
    s = _s;
}

Plane::Plane(const Plane &plane)
{
    p = plane.p;
    e = plane.e;
    c = plane.c;

    n = plane.n;
    s = plane.s;
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