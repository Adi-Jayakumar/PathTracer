#include "triangle.h"
#include <cmath>
#include <limits>

Triangle::Triangle(Vec _p1, Vec _p2, Vec _p3)
{
    p1 = _p1;
    p2 = _p2;
    p3 = _p3;
    n = Vec::Cross(p3 - p1, p2 - p1).Norm();
}

bool Triangle::Intersect(Ray &r, double &hit)
{
    Vec edge1 = p2 - p1;
    Vec edge2 = p3 - p1;
    Vec h = Vec::Cross(r.d, edge2);
    double a = Vec::Dot(edge1, h);
    if (fabs(a) < 1e-7)
        return false;
    Vec s = r.o - p1;
    double u = Vec::Dot(s, h) / a;
    if (u < 0 || u > 1)
        return false;
    Vec q = Vec::Cross(s, edge1);
    double v = Vec::Dot(r.d, q) / a;
    if (v < 0 || u + v > 1)
        return false;
    double t = Vec::Dot(edge2, q) / a;
    if (t > PTUtility::EPSILON)
    {
        hit = t;
        return true;
    }
    return false;
}

Vec Triangle::Normal(Vec &x)
{
    return n;
}

void Triangle::Translate(Vec &x)
{
    p1 = p1 + x;
    p2 = p2 + x;
    p3 = p3 + x;
}

bool Triangle::IsOnSkin(Vec &x)
{
    Vec AB = p2 - p1;
    Vec BC = p3 - p2;
    Vec CA = p1 - p3;

    Vec AP = x - p1;
    Vec BP = x - p2;
    Vec CP = x - p3;

    Vec u = Vec::Cross(AB, AP);
    Vec v = Vec::Cross(BC, BP);
    Vec w = Vec::Cross(CA, CP);

    double d1 = Vec::Dot(n, u);
    double d2 = Vec::Dot(n, v);
    double d3 = Vec::Dot(n, w);

    return (d1 > 0 && d2 > 0 && d3 > 0) || (d1 < 0 && d2 < 0 && d3 < 0);
}

double Triangle::FarSolution(Ray &ray)
{
    Vec edge1 = p2 - p1;
    Vec edge2 = p3 - p1;
    Vec h = Vec::Cross(ray.d, edge2);
    double a = Vec::Dot(edge1, h);
    if (fabs(a) < 1e-7)
        return std::numeric_limits<double>::max();
    Vec s = ray.o - p1;
    double u = Vec::Dot(s, h) / a;
    if (u < 0 || u > 1)
        return std::numeric_limits<double>::max();
    Vec q = Vec::Cross(s, edge1);
    double v = Vec::Dot(ray.d, q) / a;
    if (v < 0 || u + v > 1)
        return std::numeric_limits<double>::max();
    double t = Vec::Dot(edge2, q) / a;
    if (t > PTUtility::EPSILON)
        return t;
    return std::numeric_limits<double>::max();
}