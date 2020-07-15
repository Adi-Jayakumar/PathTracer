#include <cmath>
#include <limits>
#include "sphere.h"

Sphere::Sphere(double _r, Vec _p, Vec _e, Vec _c, Surface _s)
{

    r = _r;
    p = _p;
    e = _e;
    c = _c;
    s = _s;
}

double Sphere::Intersect(Ray &ray)
{
    double A = Vec::Dot(ray.d, ray.d);
    double B = 2 * Vec::Dot(ray.o, ray.d) - 2 * Vec::Dot(p, ray.d);
    double C = Vec::Dot(ray.o, ray.o) + Vec::Dot(p, p) - 2 * Vec::Dot(ray.o, p) - r * r;

    return PTUtility::SolveQuadratic(A, B, C);
};

Vec Sphere::Normal(Vec &v)
{
    return (v - p) / r;
}

void Sphere::Translate(Vec &x)
{
    p = p + x;
}