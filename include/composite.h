#pragma once
#include "shape.h"
#include <memory>

enum class SetOp
{
    AND,
    OR,
    SUB,
};

class Composite : public Shape
{
public:
    std::shared_ptr<Shape> lhs;
    std::shared_ptr<Shape> rhs;
    SetOp op;
    Composite(std::shared_ptr<Shape> _lhs, std::shared_ptr<Shape> _rhs, SetOp _op);

    std::shared_ptr<Composite> operator|(std::shared_ptr<Shape> lhs, std::shared_ptr<Shape> rhs);
    

    bool Intersect(Ray &ray, double &hit) override;
    Vec Normal(Vec &x) override;
    void Translate(Vec &x) override;
    bool IsOnSkin(Vec &x) override;
};
