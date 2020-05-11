#include <iostream>
#include <limits>
#include <cmath>
#include "scene.h"
#include "image.h"

void Scene::AddSolid(Solid *s)
{
    objects.emplace_back(s);
}

void Scene::AddCamera(Camera c)
{
    cameras.emplace_back(c);
}

HitRecord Scene::ClosestIntersection(Ray r)
{
    // Assume no hit initally
    double recordT = std::numeric_limits<double>::max();
    int idMin = -1;

    // iterate through and check for a hit
    for (long unsigned i = 0; i < objects.size(); ++i)
    {
        double t = objects[i]->Intersect(r);

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

Vec Scene::PixelColour(Ray r, int depth)
{
    // if it has bounced enough times return black
    if (depth > PTMath::MaxDepth)
        return Vec(0, 0, 1);

    HitRecord rec = ClosestIntersection(r);

    // if no sphere was hit return black
    if (rec.id == -1)
    {
        return Vec();
    }

    // the object that was hit
    Solid *hit = objects[rec.id];
    // the hit-point
    Vec hitPt = r.o + r.d * rec.t;
    // the surface normal
    Vec normal = hit->Normal(hitPt);
    // the normal that faces toward the ray
    Vec rayNormal = Vec::Dot(normal, r.d) < 0 ? normal : normal * -1;

    Vec albedo = hit->c;

    // probability of terminating early
    double finProb = PTMath::Luma(albedo);
    if (++depth > 5)
    {
        if (PTMath::Random() < finProb)
            albedo = albedo / finProb;
        else
            return hit->e;
    }

    if (hit->s == DIFF)
    {
        double phi = 2 * PTMath::PI * PTMath::Random();
        double r2 = PTMath::Random();
        double sinTheta = sqrt(r2);
        double cosTheta = sqrt(1 - r2);

        // building basis for R3 of perpendicular vectors around the normal facing outwards
        Vec w = rayNormal.Norm();
        // Vec u = (Vec::Cross(fabs(w.x) > .1 ? Vec(0, 1, 0) : Vec(1, 0, 0), w)).Norm();
        Vec u = Vec(rayNormal.z, rayNormal.z, -rayNormal.x - 2 * rayNormal.y).Norm();
        Vec v = Vec::Cross(u, w).Norm();

        Vec newDir = (u * cos(phi) * sinTheta + v * sin(phi) * sinTheta + w * cosTheta).Norm();
        return hit->e + albedo * PixelColour(Ray(hitPt, newDir), depth);
    }
    else if (hit->s == SPEC)
    {
        // std::cout <<rayNormal << std::endl;
        Vec reflectedDir = r.d - rayNormal * 2 * Vec::Dot(rayNormal, r.d);
        return hit->e + albedo * PixelColour(Ray(hitPt, reflectedDir), depth);
    }
    else if (hit->s == REFR)
    {
        Ray reflectedRay = Ray(hitPt, r.d - rayNormal * 2 * Vec::Dot(rayNormal, r.d));

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
            return hit->e + albedo * PixelColour(reflectedRay, depth);

        Vec rerfactedDir = (r.d * netN - normal * ((isInto ? 1 : -1) * (cosTheta * cosTheta + sqrt(cosTheta2Sqr)))).Norm();
        // Vec rerfactedDir = (r.d * netN - rayNormal * (cosTheta * cosTheta + sqrt(cosTheta2Sqr)))).Norm();

        // approximating reflection and refraction weights
        double a = n2 - n1;
        double b = n1 + n2;
        double R0 = (a * a) / (b * b);

        double cosTheta2 = Vec::Dot(rerfactedDir, normal);
        double c = 1 - (isInto ? -cosTheta : cosTheta2);
        double refl = R0 + (1 - R0) * c * c * c * c;
        double refr = 1 - refl;

        double P = 0.25 + 0.5 * refl;

        double reflectionWeight = refl / P;
        double refractionWeight = refr / P;

        if (depth < 3)
            return hit->e + albedo * (PixelColour(reflectedRay, depth) * refl + PixelColour(Ray(hitPt, rerfactedDir), depth) * refr);
        else
        {
            if (PTMath::Random() < P)
                return hit->e + albedo * PixelColour(reflectedRay, depth) * reflectionWeight;
            else
                return hit->e + albedo * PixelColour(Ray(hitPt, rerfactedDir), depth) * refractionWeight;
                }
    }
    return Vec();
}



void Scene::TakePicture(int index)
{
    Vec image[PTMath::W * PTMath::H];
    Image im = Image(PTMath::W, PTMath::H, index);
    Ray r;

#pragma omp parallel for schedule(dynamic, 1) private(r)
    // image rows
    for (int i = 0; i < PTMath::H; i++)
    {
        // image cols
        for (int j = 0; j < PTMath::W; j++)
        {
            Vec c;
            // sub pixel rows
            for (int sy = 0; sy < PTMath::SubPixSize; sy++)
            {
                // sub pixel cols
                for (int sx = 0; sx < PTMath::SubPixSize; sx++)
                {
                    // sampling the pixel
                    for (int s = 0; s < PTMath::NumSamps; s++)
                    {
                        r = cameras[index].GenerateRay(i, j, sx, sy);
                        c = c + PixelColour(r, 0);
                    }
                }
            }
            image[i * PTMath::W + j] = c / ((double)PTMath::NumSamps * PTMath::SubPixSize * PTMath::SubPixSize);
        }
    }

    im.Set(image);
}
