#pragma once
#include <memory>
#include "ray.h"
#include "shape.h"
#include "ptutility.h"

enum class Surface
{
    DIFF,
    SPEC,
    SPECGLOSS,
    REFRGLOSS,
    REFR,
};

struct Solid
{
    std::shared_ptr<Shape> shape;
    Vec e, c;
    Surface s;
    Solid(std::shared_ptr<Shape> _shape, Vec _e, Vec _c, Surface _s);
};