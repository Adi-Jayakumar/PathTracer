#include <cmath>
#include <limits>
#include "triangle.h"

Triangle::Triangle(Vec _p1, Vec _p2, Vec _p3, Vec _e, Vec _c, Surface _s)
{
    p1 = _p1;
    p2 = _p2;
    p3 = _p3;
    p = (p1 + p2 + p3) / 3;
    e = _e;
    c = _c;
    s = _s;
    n = Vec::Cross(p2 - p1, p3 - p1).Norm();
}

double Triangle::Intersect(Ray &r)
{
    Vec edge1 = p2 - p1;
    Vec edge2 = p3 - p1;
    Vec h = Vec::Cross(r.d, edge2);
    double a = Vec::Dot(edge1, h);
    if (fabs(a) < 1e-7)
        return std::numeric_limits<double>::max();
    Vec s = r.o - p1;
    double u = Vec::Dot(s, h) / a;
    if (u < 0 || u > 1)
        return std::numeric_limits<double>::max();
    Vec q = Vec::Cross(s, edge1);
    double v = Vec::Dot(r.d, q) / a;
    if (v < 0 || u + v > 1)
        return std::numeric_limits<double>::max();
    double t = Vec::Dot(edge2, q) / a;
    if (t > PTUtility::EPSILON)
        return t;
    return std::numeric_limits<double>::max();
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
    p = p + x;
}