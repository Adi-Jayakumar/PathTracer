#include "plane.h"
#include "ptmath.h"
#include <limits>

Plane::Plane(Vec _n, Vec _p, Vec _e, Vec _c, Surface _s)
{
    p = _p;
    e = _e;
    c = _c;

    n = _n;
    s = _s;
}

double Plane::Intersect(Ray &ray, double tMin)
{
    double det = Vec::Dot(ray.d, n);
    if (det == 0)
        return std::numeric_limits<double>::max();

    double t = Vec::Dot(n, p - ray.o) / det;
    if (t < tMin)
        return std::numeric_limits<double>::max();
    return t;
}

Vec Plane::Normal(Vec &x)
{
    return n;
}