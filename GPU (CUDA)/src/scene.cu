#include "image.h"
#include "scene.h"
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

__device__ Ray Scene::GenerateDiffuseRay(Vec &hitPt, Vec &normal, curandState &state)
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

__device__ Ray Scene::GenerateSpecularRay(Vec &hitPt, Vec &rayNormal, Vec &dir)
{
    Vec reflectedDir = dir - rayNormal * 2 * Vec::Dot(rayNormal, dir);
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
    Vec awayMask = Vec(1, 1, 1);
    HitRecord rec;

    for (int i = 0; i < PTUtility::MaxDepth; i++)
    {

        rec = Scene::ClosestIntersection(r, objects, nObj);
        if (rec.id == -1)
            return Vec();

        // pointer to the object that was hit
        Solid *hit = objects[rec.id];
        // point that was hit
        Vec hitPt = r.o + r.d * rec.t;
        // outward normal at that point
        Vec normal = hit->shape->Normal(hitPt);
        // normal that points in the direction of the ray
        Vec rayNormal = Vec::Dot(normal, r.d) < 0 ? normal : normal * -1;

        awayColour = awayColour + hit->e * awayMask;

        if (hit->s == Surface::DIFF)
        {
            r = Scene::GenerateDiffuseRay(hitPt, rayNormal, state);
            awayMask = awayMask * Vec::Dot(r.d, rayNormal);
            awayMask = awayMask * hit->c;
        }
        else if (hit->s == Surface::SPEC || hit->s == Surface::SPECGLOSS)
        {
            Ray cand = Scene::GenerateSpecularRay(hitPt, normal, r.d);
            awayMask = awayMask * hit->c;
            if (hit->s == Surface::SPEC)
                r = cand;
            else
                r = Ray(hitPt, Scene::Jitter(cand.d, 0.125, state));
        }
        else if (hit->s == Surface::REFR || hit->s == Surface::REFRGLOSS)
        {
            Ray reflectedRay = Ray(hitPt, r.d - normal * 2 * Vec::Dot(normal, r.d));

            // is the ray going into the object or out?
            bool isInto = Vec::Dot(normal, rayNormal) > 0;

            // refractive indices
            double n1 = 1;
            double n2 = 1.5;
            double netN = isInto ? n1 / n2 : n2 / n1;

            // cosines of the angles
            double cosTheta = Vec::Dot(r.d, rayNormal);
            double cosTheta2Sqr = 1 - netN * netN * (1 - cosTheta * cosTheta);

            // total internal reflection
            if (cosTheta2Sqr < 0)
            {
                r = reflectedRay;
                if(hit->s == Surface::REFRGLOSS)
                    r.d = Scene::Jitter(r.d, 0.125, state);
            }
            else
            {
                Vec refractedDir = (r.d * netN - normal * ((isInto ? 1 : -1) * (cosTheta * cosTheta + sqrt(cosTheta2Sqr)))).Norm();
                Ray refractedRay = Ray(hitPt, refractedDir);
                // Vec refractedDir = (r.d * netN - rayNormal * (cosTheta * cosTheta + sqrt(cosTheta2Sqr)))).Norm();

                // approximating reflection and refraction weights
                double a = n2 - n1;
                double b = n1 + n2;
                double R0 = (a * a) / (b * b);

                double cosTheta2 = Vec::Dot(refractedDir, normal);
                double c = 1 - (isInto ? -cosTheta : cosTheta2);
                double refl = R0 + (1 - R0) * c * c * c * c * c;
                double refr = 1 - refl;

                double P = 0.25 + 0.5 * refl;

                double reflectionWeight = refl / P;
                double refractionWeight = refr / (1 - P);

                if (PTUtility::Random(state) < P)
                {
                    awayMask = awayMask * reflectionWeight;
                    r = reflectedRay;
                    if(hit->s == Surface::REFRGLOSS)
                        r.d = Scene::Jitter(r.d, 0.125, state);
                        
                }
                else
                {
                    awayMask = awayMask * refractionWeight;
                    r = refractedRay;
                    if (hit->s == Surface::REFRGLOSS)
                        r.d = Scene::Jitter(r.d, 0.03125, state);
                }
            }
        }
    }

    return awayColour;
}