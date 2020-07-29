#include <cmath>
#include <limits>
#include "sphere.h"

Sphere::Sphere(double _r, Vec _c)
{

    r = _r;
    c = _c;
}

double Sphere::Intersect(Ray &ray)
{
    double A = Vec::Dot(ray.d, ray.d);
    double B = 2 * Vec::Dot(ray.o, ray.d) - 2 * Vec::Dot(c, ray.d);
    double C = Vec::Dot(ray.o, ray.o) + Vec::Dot(c, c) - 2 * Vec::Dot(ray.o, c) - r * r;

    return PTUtility::SolveQuadratic(A, B, C);
};

Vec Sphere::Normal(Vec &v)
{
    return (v - c) / r;
}

void Sphere::Translate(Vec &x)
{
    c = c + x;
}