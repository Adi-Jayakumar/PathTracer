#include "composite.h"

Composite::Composite(std::shared_ptr<Shape> _lhs, std::shared_ptr<Shape> _rhs, SetOp _op)
{
    lhs = _lhs;
    rhs = _rhs;
    op = _op;
}

std::shared_ptr<Composite> operator|(std::shared_ptr<Shape> lhs, std::shared_ptr<Shape> rhs)
{
    return std::make_shared<Composite>(lhs, rhs, SetOp::OR);
}

std::shared_ptr<Composite> operator&(std::shared_ptr<Shape> lhs, std::shared_ptr<Shape> rhs)
{
    return std::make_shared<Composite>(lhs, rhs, SetOp::AND);
}

std::shared_ptr<Composite> operator-(std::shared_ptr<Shape> lhs, std::shared_ptr<Shape> rhs)
{
    return std::make_shared<Composite>(lhs, rhs, SetOp::SUB);
}

bool Composite::Intersect(Ray &ray, double &hit)
{
    switch (op)
    {
    case SetOp::OR:
    {
        double t1, t2;
        if (lhs->Intersect(ray, t1) || rhs->Intersect(ray, t2))
        {
            hit = std::min(t1, t2);
            return true;
        }
        return false;
    }
    case SetOp::AND:
    {
        double t1, t2;
        if (lhs->Intersect(ray, t1) && rhs->Intersect(ray, t2))
        {
            hit = std::max(t1, t2);
            return true;
        }
        return false;
    }
    default:
    {
        hit = std::numeric_limits<double>::max();
        return false;
    }
    }
}

Vec Composite::Normal(Vec &x)
{
    switch (op)
    {
    case SetOp::AND:
    case SetOp::OR:
    {
        if (lhs->IsOnSkin(x))
            return lhs->Normal(x);
        else
            return rhs->Normal(x);
        break;
    }
    default:
    {
        if(lhs->IsOnSkin(x))
            return lhs->Normal(x);
        else
            return rhs->Normal(x) * -1;
        break;
    }
    }
}

void Composite::Translate(Vec &x)
{
    lhs->Translate(x);
    rhs->Translate(x);
}

bool Composite::IsOnSkin(Vec &x)
{
    return lhs->IsOnSkin(x) || rhs->IsOnSkin(x);
}