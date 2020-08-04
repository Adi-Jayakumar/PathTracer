#include "composite.h"

Composite::Composite(std::shared_ptr<Shape> _lhs, std::shared_ptr<Shape> _rhs, SetOp _op)
{
    lhs = _lhs;
    rhs = _rhs;
    op = _op;
}

std::shared_ptr<Shape> operator&(std::shared_ptr<Shape> lhs, std::shared_ptr<Shape> rhs)
{
    return std::make_shared<Composite>(lhs, rhs, SetOp::AND);
}

std::shared_ptr<Shape> operator|(std::shared_ptr<Shape> lhs, std::shared_ptr<Shape> rhs)
{
    return std::make_shared<Composite>(lhs, rhs, SetOp::OR);
}

std::shared_ptr<Shape> operator-(std::shared_ptr<Shape> lhs, std::shared_ptr<Shape> rhs)
{
    return std::make_shared<Composite>(lhs, rhs, SetOp::SUB);
}

bool Composite::Intersect(Ray &ray, double &hit)
{
    switch (op)
    {
    case SetOp::AND:
    {
        double t1 = std::numeric_limits<double>::max();
        double t2 = std::numeric_limits<double>::max();
        if (lhs->Intersect(ray, t1) && rhs->Intersect(ray, t2))
        {
            hit = std::max(t1, t2);
            return true;
        }
        hit = std::numeric_limits<double>::max();
        return false;
    }
    case SetOp::OR:
    {
        double t1 = std::numeric_limits<double>::max();
        double t2 = std::numeric_limits<double>::max();

        bool hitLHS = lhs->Intersect(ray, t1);
        bool hitRHS = rhs->Intersect(ray, t2);

        if (hitLHS || hitRHS)
        {
            hit = std::min(t1, t2);
            return true;
        }
        hit = std::numeric_limits<double>::max();
        return false;
    }
    default:
    {
        double t1 = std::numeric_limits<double>::max();
        double t2 = std::numeric_limits<double>::max();
        if (lhs->Intersect(ray, t1))
        {
            if (!rhs->Intersect(ray, t2))
            {
                hit = t1;
                return true;
            }
            else
            {
                hit = rhs->FarSolution(ray);
                return true;
            }
        }
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
        if (lhs->IsOnSkin(x))
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

double Composite::FarSolution(Ray &ray)
{
    double hit = 0;
    if(lhs->Intersect(ray,hit) && !rhs->Intersect(ray, hit))
        return lhs->FarSolution(ray);
    else if (!lhs->Intersect(ray, hit) && rhs->Intersect(ray, hit))
        return rhs->FarSolution(ray);
}