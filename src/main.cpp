#include <iostream>
#include <cmath>
#include "vector.h"
#include "scene.h"
#include "sphere.h"
#include "triangle.h"
#include "plane.h"
#include "image.h"
#include "hitrecord.h"
#include "camera.h"
#include "ptmath.h"

using namespace std;

// RUN: clear && make && ./bin/main
int main()
{



    Camera CamUp = Camera(10, PTMath::W, PTMath::H, Vec(0, 0., 0), Vec(0, 0, 1.), Vec(0, 1, 0), PTMath::PI / 3.0);
    Camera CamDown = Camera(10, PTMath::W, PTMath::H, Vec(0, 0., 0), Vec(0, 0, 1.), Vec(0, -1, 0), PTMath::PI / 3.0);

    Triangle t = Triangle(Vec(-1, 0, 1), Vec(0, 2, 0), Vec(1, 0, 1), Vec(), Vec(1, 0, 0), DIFF);
    Sphere s = Sphere(1, Vec(0,0, -1.01), Vec(1,1,1), Vec(), DIFF);

    Scene scene;
    scene.AddCamera(CamUp);
    scene.AddCamera(CamDown);
    scene.AddSolid(&t);
    scene.AddSolid(&s);

    scene.TakePicture(1);



    return 0;
}
