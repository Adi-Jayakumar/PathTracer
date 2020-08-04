#pragma once
#include "ptutility.h"
#include "ray.h"
#include <memory>

struct Shape
{
    // thses 2 are needed for path tracing
    virtual bool Intersect(Ray &ray, double &hit) = 0;
    virtual Vec Normal(Vec &x) = 0;
    
    virtual void Translate(Vec &x) = 0;

    // these 2 are used in Composite
    virtual bool IsOnSkin(Vec &x) = 0;
    virtual double FarSolution(Ray &ray) = 0;

    friend std::shared_ptr<Shape> operator&(std::shared_ptr<Shape> lhs, std::shared_ptr<Shape> rhs);
    friend std::shared_ptr<Shape> operator|(std::shared_ptr<Shape> lhs, std::shared_ptr<Shape> rhs);
    friend std::shared_ptr<Shape> operator-(std::shared_ptr<Shape> lhs, std::shared_ptr<Shape> rhs);
};