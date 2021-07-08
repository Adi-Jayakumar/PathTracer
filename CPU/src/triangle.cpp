#include "triangle.h"

Triangle::Triangle(Vec _p1, Vec _p2, Vec _p3)
{
    p1 = _p1;
    p2 = _p2;
    p3 = _p3;
    n = Vec::Cross(p3 - p1, p2 - p1).Norm();
}

bool Triangle::Intersect(Ray &r, double &hit)
{
    hit = std::numeric_limits<double>::max();
    Vec edge1 = p2 - p1;
    Vec edge2 = p3 - p1;
    Vec h = Vec::Cross(r.d, edge2);
    double a = Vec::Dot(edge1, h);

    bool didHit = false;
    double t = std::numeric_limits<double>::max();

    if (fabs(a) > 1e-7)
    {
        Vec s = r.o - p1;
        double u = Vec::Dot(s, h) / a;
        if (u >= 0 && u <= 1)
        {
            Vec q = Vec::Cross(s, edge1);
            double v = Vec::Dot(r.d, q) / a;
            if (v >= 0 && u + v <= 1)
            {
                t = Vec::Dot(edge2, q) / a;
                if (t > PTUtility::EPSILON)
                {
                    hit = t;
                    didHit = true;
                }
            }
        }
    }
    return didHit;
}

Vec Triangle::Normal(Vec &)
{
    return n;
}

void Triangle::Translate(Vec &x)
{
    p1 = p1 + x;
    p2 = p2 + x;
    p3 = p3 + x;
}