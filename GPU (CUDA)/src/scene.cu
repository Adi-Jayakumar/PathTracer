#include "image.cuh"
#include "scene.cuh"
#include <curand_kernel.h>

__device__ HitRecord Scene::ClosestIntersection(Ray r, Solid **objects, int nObj)
{
    // Assume no hit initally
    double recordT = PTUtility::INF;
    int idMin = -1;
    double t;

    // iterate through and check for a hit
    for (long unsigned i = 0; i < nObj; i++)
    {
        t = PTUtility::INF;
        objects[i]->shape->Intersect(r, t);

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


__device__ Ray Scene::GenerateDiffuseRay(Vec& hitPt, Vec& normal, curandState &state)
{
    double phi = 2 * PTUtility::PI * PTUtility::Random(state);
    double r2 = PTUtility::Random(state);

    double sinTheta = sqrt(r2);
    double cosTheta = sqrt(1 - r2);

    // building basis for R3 of perpendicular vectors around the normal facing outwards
    Vec w = normal.Norm();
    // Vec u = (Vec::Cross(fabs(w.x) > .1 ? Vec(0, 1, 0) : Vec(1, 0, 0), w)).Norm();
    Vec u = Vec(normal.z, normal.z, -normal.x - 2 * normal.y).Norm();
    Vec v = Vec::Cross(u, w).Norm();

    Vec newDir = (u * cos(phi) * sinTheta + v * sin(phi) * sinTheta + w * cosTheta).Norm();
    return Ray(hitPt, newDir);
}

__device__ Ray Scene::GenerateSpecularRay(Vec &hitPt, Vec &normal, Vec &dir)
{
    Vec reflectedDir = dir - normal * 2 * Vec::Dot(normal, dir);
    return Ray(hitPt, reflectedDir);
}

__device__ Vec Scene::Jitter(Vec x, double r, curandState &state)
{
    Vec w = x.Norm();
    Vec u = Vec(w.z, w.z, -w.x - 2 * w.y).Norm();
    Vec v = Vec::Cross(u, w).Norm();

    double rad = r * sqrt(PTUtility::Random(state));
    double theta = PTUtility::Random(state) * 2 * PTUtility::PI;

    double xPrime = rad * cos(theta);
    double yPrime = rad * sin(theta);

    return (u * xPrime + v * yPrime + w).Norm();
}

__device__ Vec Scene::RayColour(Ray r, Solid **objects, curandState &state, int nObj)
{

    // translating recursion of CPU pathtracer into a loop since
    // call-stack sizes on GPU are much smaller and so origin code
    // would just overflow

    Vec awayColour = Vec(0, 0, 0);
    Vec mask = Vec(1, 1, 1);
    HitRecord awayRec;
    double awayWeight = 1;

    bool isTrans = false;

    Ray transRay = Ray();
    Vec transColour = Vec(0, 0, 0);
    Vec transMask = Vec(1, 1, 1);
    HitRecord transRec;
    double transWeight = 0;

    for (int i = 0; i < PTUtility::MaxDepth; i++)
    {

        awayRec = Scene::ClosestIntersection(r, objects, nObj);
        transRec = Scene::ClosestIntersection(transRay, objects, nObj);

        if (awayRec.id == -1)
            return Vec();

        // pointer to the object that was hit
        Solid *hit = objects[awayRec.id];
        // point that was hit
        Vec hitPt = r.o + r.d * awayRec.t;
        // outward normal at that point
        Vec normal = hit->shape->Normal(hitPt);
        // normal that points in the direction of the ray
        Vec rayNormal = Vec::Dot(normal, r.d) < 0 ? normal : normal * -1;

        awayColour = awayColour + mask * hit->e;
        mask = mask * hit->c;

        if (hit->s == Surface::DIFF)
        {
            r = Scene::GenerateDiffuseRay(hitPt, rayNormal, state);
            mask = mask * Vec::Dot(r.d, rayNormal);
        }
        else if (hit->s == Surface::SPEC || hit->s == Surface::SPECGLOSS)
        {
            Ray cand = Scene::GenerateSpecularRay(hitPt, normal, r.d);
            if(hit->s == Surface::SPEC)
                r = cand;
            else
                r = Ray(hitPt, Scene::Jitter(cand.d, 0.125, state));

        }
        // else if (hit->s == Surface::REFR || hit->s == Surface::REFRGLOSS)
        // {
        //     isTrans = true;
            
        // }
    }

    return awayColour * awayWeight + transColour * transWeight;
}