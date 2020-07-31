#include "triangle.h"
#include <cmath>
#include <limits>

Triangle::Triangle(Vec _p1, Vec _p2, Vec _p3)
{
    p1 = _p1;
    p2 = _p2;
    p3 = _p3;
    n = Vec::Cross(p2 - p1, p3 - p1).Norm();
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