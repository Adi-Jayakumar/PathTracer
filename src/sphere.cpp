#include "sphere.h"
#include <cmath>
#include <limits>

Sphere::Sphere(double _r, Vec _c)
{

    r = _r;
    c = _c;
}

bool Sphere::Intersect(Ray &ray, double &hit)
{
    double A = Vec::Dot(ray.d, ray.d);
    double B = 2 * Vec::Dot(ray.o, ray.d) - 2 * Vec::Dot(c, ray.d);
    double C = Vec::Dot(ray.o, ray.o) + Vec::Dot(c, c) - 2 * Vec::Dot(ray.o, c) - r * r;

    hit = PTUtility::SolveQuadratic(A, B, C);
    if (hit == std::numeric_limits<double>::max())
        return false;
    else
        return true;
};

Vec Sphere::Normal(Vec &v)
{
    return (v - c) / r;
}

void Sphere::Translate(Vec &x)
{
    c = c + x;
}