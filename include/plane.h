#pragma once
#include "shape.h"

struct Plane : Shape
{
    Vec n; // normal
    Vec p;
    Plane(Vec _n, Vec _p);
    Plane(const Plane &plane);
    bool Intersect(Ray &ray, double &hit) override;
    Vec Normal(Vec &x) override;
    void Translate(Vec &x) override;
    bool IsOnSkin(Vec &x) override;
    double FarSolution(Ray & ray) override;
};