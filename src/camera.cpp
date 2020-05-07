#include <cmath>
#include <iostream>
#include "ptmath.h"
#include "camera.h"

Camera::Camera(double _worldW, int _nPixelsX, int _nPixelsY, Vec _loc, Vec _forward, Vec _up, double _hFOV)
{
    worldW = _worldW;
    nPixelsX = _nPixelsX;
    nPixelsY = _nPixelsY;
    worldH = worldW * (double)nPixelsY / nPixelsX;
    loc = _loc;
    forward = _forward.Norm();
    up = _up.Norm();
    hFOV = _hFOV;
    right = Vec::Cross(up, forward).Norm();
    focus = _loc - forward * (worldW / (2 * tan(hFOV / 2)));

    // std::cout << "worldW " << worldW << std::endl;
    // std::cout << "worldH " << worldH << std::endl;
    // std::cout << "nPixelsX " << nPixelsX << std::endl;
    // std::cout << "nPixelsY " << nPixelsY << std::endl;
    // std::cout << "forward " << forward << std::endl;
    // std::cout << "up " << up << std::endl;
    // std::cout << "hFOV " << hFOV << std::endl;
    // std::cout << "right " << right << std::endl;
    // std::cout << "d " << worldW / (2 * tan(hFOV / 2)) << std::endl;
    // std::cout << "focus" << focus << std::endl;
};

Ray Camera::GenerateRay(int i, int j, int sx, int sy)
{
    /*
    0 <= j < nPixelsX, 0 <= i < nPixelsY
    o----------------> j increasing
    |   o--- 
    |    \ |
    |     \|
    |      c
    |
    |
    |
    v
    i increasing

    c is the centre of the screen, o is some point on it

    The point o is at (j, i) in the image and once i and j are normalised to be between 0 and 1,
    we can see that with respect to c, o becomes (j - 0.5, 0.5 - i) in a world where the top of 
    the screen is up and the right is right
    */
    double r1 = 2 * PTMath::Random();
    double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
    double r2 = 2 * PTMath::Random();
    double dy = r1 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
    Vec deltaRight = right * worldW * (((double)j + (sx + 0.5 + dx) / 2) / nPixelsX - 0.5);
    Vec deltaUp = up * worldH * (0.5 - ((double)i + (sy + 0.5 + dy) / 2) / nPixelsY);
    Vec rayOr = deltaUp + deltaRight + loc;
    return Ray(rayOr, (rayOr - focus).Norm());
}

void Camera::AddToScene(Sphere s)
{
    scene.AddSphere(s);
}

Vec Camera::PixelColour(Ray r, int depth)
{
    // if it has bounced enough times return black
    if (depth > PTMath::MaxDepth)
        return Vec(0, 0, 1);

    HitRecord rec = scene.ClosestIntersection(r, 1e-5);

    // if no sphere was hit return black
    if (rec.id == -1)
    {
        return Vec();
    }

    // the object that was hit
    Sphere &hit = scene.objects[rec.id];
    // the hit-point
    Vec hitPt = r.o + r.d * rec.t;
    // the surface normal
    Vec normal = hit.Normal(hitPt);
    // the normal that faces toward the ray
    Vec rayNormal = Vec::Dot(normal, r.d) < 0 ? normal : normal * -1;

    Vec albedo = hit.c;

    // probability of terminating early
    double finProb = PTMath::Luma(albedo);
    if (++depth > 5)
    {
        if (PTMath::Random() < finProb)
            albedo = albedo / finProb;
        else
            return hit.e;
    }

    if (hit.s == DIFF)
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
        return hit.e + albedo * PixelColour(Ray(hitPt, newDir), depth);
    }
    else if (hit.s == SPEC)
    {
        // std::cout <<rayNormal << std::endl;
        Vec reflectedDir = r.d - rayNormal * 2 * Vec::Dot(rayNormal, r.d);
        return hit.e + albedo * PixelColour(Ray(hitPt, reflectedDir), depth);
    }
    // else if (hit.s == REFR)
    // {
    //     Ray reflectedRay = Ray(hitPt, r.d - normal * 2 * Vec::Dot(normal, r.d));
    //     bool isInto = Vec::Dot(normal, rayNormal) > 0;
    //     double n1 = 1;
    //     double n2 = 1.5;
    //     double netN = isInto ? n1/n2 : n2/n1;
    //     double cosTheta = Vec::Dot(r.d, rayNormal);
    //     double cosTheta2Sqr = 1 - netN * netN * (1 - cosTheta * cosTheta);
    //     if(cosTheta2Sqr < 0)
    // }
    return Vec();
}