#pragma once
#include "ray.h"
#include "shape.h"
#include "vector.h"

struct Triangle : Shape
{
    Vec p1, p2, p3, n;
    Triangle(Vec _p1, Vec _p2, Vec _p3);
    bool Intersect(Ray &ray, double &hit, std::shared_ptr<std::pair<double, double>>values = nullptr) override;
    Vec Normal(Vec &x) override;
    void Translate(Vec &x) override;
    bool IsOnSkin(Vec &x) override;
};