#include "solid.h"

enum class SetOp
{
    AND,
    OR,
    SUB,
};

class Composite : public Solid
{
public:
    Composite *lhs;
    Composite *rhs;
    SetOp op;
    virtual double Intersect(Ray &ray) = 0;
    virtual Vec Normal(Vec &x) = 0;
    virtual void Translate(Vec &x) = 0;
};
