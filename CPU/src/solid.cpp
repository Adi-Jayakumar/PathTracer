#include "solid.h"
Solid::Solid(std::shared_ptr<Shape> _shape, Vec _e, Vec _c, Surface _s)
{
    shape = _shape;
    e = _e;
    c = _c;
    s = _s;
}