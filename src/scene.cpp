#include <iostream>
#include "scene.h"

void Scene::AddSphere(Solid *s)
{
    objects.emplace_back(s);
}

HitRecord Scene::ClosestIntersection(Ray r, double tMin)
{
    // Assume no hit initally
    double recordT = std::numeric_limits<double>::max();
    int idMin = -1;

    // iterate through and check for a hit
    for (long unsigned i = 0; i < objects.size(); ++i)
    {
        double t = objects[i]->Intersect(r, tMin);

        // if hit then set the "record" values accordingly
        if (t < recordT)
        {
            recordT = t;
            idMin = i;
        }
    }

    // return a hitrecord with the details
    return HitRecord(recordT, idMin);
}