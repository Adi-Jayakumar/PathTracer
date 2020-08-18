#pragma once
#include "shape.h"
#include <cmath>
#include <limits>

struct Plane : Shape
{
    Vec n; // normal
    Vec p;
    Plane(Vec _n, Vec _p);
    bool Intersect(Ray &ray, double &hit, std::shared_ptr<std::pair<double, double>> values = nullptr) override;
    Vec Normal(Vec &x) override;
    void Translate(Vec &x) override;
    bool IsOnSkin(Vec &x) override;
};