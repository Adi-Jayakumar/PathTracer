#pragma once
#include "ptutility.h"
#include "ray.h"
#include "shape.h"
#include "vector.h"
#include <cmath>
#include <limits>
#include <memory>

struct Sphere : Shape
{
    double r;
    Vec c;
    Sphere(double _r, Vec _c);
    bool Intersect(Ray &ray, double &hit) override;
    Vec Normal(Vec &v) override;
    void Translate(Vec &x) override;
};