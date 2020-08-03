#pragma once
#include "ray.h"
#include "shape.h"
#include "vector.h"

class Triangle : public Shape
{
public:
    Vec p1, p2, p3, n;
    Triangle(Vec _p1, Vec _p2, Vec _p3);
    bool Intersect(Ray &r, double &hit) override;
    Vec Normal(Vec &x) override;
    void Translate(Vec &x) override;
    bool IsOnSkin(Vec &x) override;
};