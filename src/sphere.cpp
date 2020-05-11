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

    double discriminant = B * B - 4 * A * C;

    // Checking solutions exist
    if (discriminant < 0)
        return std::numeric_limits<double>::max();

    // Finding both
    double t1 = (-B - sqrt(discriminant)) / (2 * A);
    double t2 = (-B + sqrt(discriminant)) / (2 * A);

    if (t2 <= t1)
        std::swap(t1, t2);

    // We now know that t1 < t2
    if (t2 < 0)
        return std::numeric_limits<double>::max();
    else if (t1 >= 0)
    {
        if (t1 > PTMath::EPSILON)
            return t1;
        else
            return std::numeric_limits<double>::max();
    }
    else if (t1 < 0)
    {
        if (t2 > PTMath::EPSILON)
            return t2;
        else
            return std::numeric_limits<double>::max();
    }

    return std::numeric_limits<double>::max();
};

Vec Sphere::Normal(Vec &v)
{
    return (v - p) / r;
}

void Sphere::Translate(Vec &x)
{   
    p = p + x;
}