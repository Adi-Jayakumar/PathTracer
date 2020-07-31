#include "composite.h"

Composite::Composite(std::shared_ptr<Shape> _lhs, std::shared_ptr<Shape> _rhs, SetOp _op)
{
    lhs = _lhs;
    rhs = _rhs;
    op = _op;
}

Vec Composite::Normal(Vec &x)
{
    return Vec();
}
void Composite::Translate(Vec &x)
{
    return;
}