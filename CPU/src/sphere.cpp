#include "sphere.h"


Sphere::Sphere(double _r, Vec _c)
{

    r = _r;
    c = _c;
}

bool Sphere::Intersect(Ray &ray, double &hit, std::shared_ptr<std::pair<double, double>> values)
{
    double A = Vec::Dot(ray.d, ray.d);
    double B = 2 * Vec::Dot(ray.o, ray.d) - 2 * Vec::Dot(c, ray.d);
    double C = Vec::Dot(ray.o, ray.o) + Vec::Dot(c, c) - 2 * Vec::Dot(ray.o, c) - r * r;
    bool didHit = false;

    std::pair<double, double> vals = PTUtility::SolveQuadratic(A, B, C);
    if (values != nullptr)
    {
        values->first = vals.first;
        values->second = vals.second;
    }
    hit = std::numeric_limits<double>::max();
    if (vals.first > PTUtility::EPSILON && vals.first != std::numeric_limits<double>::max())
    {
        didHit = true;
        hit = vals.first;
    }
    else if (vals.second > PTUtility::EPSILON && vals.second != std::numeric_limits<double>::max())
    {
        didHit = true;
        hit = vals.second;
    }
    return didHit;
};

Vec Sphere::Normal(Vec &v)
{
    return (v - c) / r;
}

void Sphere::Translate(Vec &x)
{
    c = c + x;
}

bool Sphere::IsOnSkin(Vec &x)
{
    return fabs((x - c).ModSq() - r * r) < PTUtility::EPSILON;
}
