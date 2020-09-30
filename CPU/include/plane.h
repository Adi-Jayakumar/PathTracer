#pragma once
#include "shape.h"
#include <memory>
#include "ptutility.h"
#include <cmath>
#include <limits>

struct Plane : Shape
{
    // n.(x - p) = 0
    Vec n; // normal
    Vec p; // location of plane
    Plane(Vec _n, Vec _p);
    bool Intersect(Ray &ray, double &hit) override;
    Vec Normal(Vec &x) override;
    void Translate(Vec &x) override;
};