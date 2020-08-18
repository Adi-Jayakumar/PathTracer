#pragma once
#include <cmath>
#include <iostream>
struct Vec
{
    double x, y, z; // x, y, z components
    Vec();
    Vec(double _x, double _y, double _z);
    friend std::ostream &operator<<(std::ostream &out, Vec const &v);
    friend Vec operator+(Vec u, Vec v);
    friend Vec operator-(Vec u, Vec v);
    friend Vec operator*(Vec u, Vec v);
    Vec operator*(double k);
    Vec operator/(double k);
    static Vec Cross(Vec u, Vec v); // cross product of 2 vectors
    double Mod(); // returns the modulus of the vector
    double ModSq(); // returns the modulus of the vector squared
    static double Dot(Vec u, Vec v); // dot product of 2 vectors
    Vec Norm(); // returns a normalised vector
};
