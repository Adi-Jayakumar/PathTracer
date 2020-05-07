#include <iostream>
#include <cmath>
#include "vector.h"
#include "sphere.h"
#include "image.h"
#include "hitrecord.h"
#include "camera.h"
#include "ptmath.h"

using namespace std;

// RUN: clear && make && ./bin/main
int main()
{

    int w = 128;
    int h = 128;
    int numSamps = 100;
    int sPixSize = 2;

    Camera cam = Camera(0.1, w, h, Vec(0,0, 25), Vec(0, 0, 1.), Vec(0, 1, 0), PTMath::PI* 0.6);

    //lights
    cam.AddToScene(Sphere(5., Vec(0, 10, 35),  Vec(1, 0, 0), Vec(), DIFF));
    cam.AddToScene(Sphere(5., Vec(8, -6, 35),  Vec(0, 1, 0), Vec(), DIFF));
    cam.AddToScene(Sphere(5., Vec(-8, -6, 35),  Vec(0, 0, 1), Vec(), DIFF));

    cam.AddToScene(Sphere(1000., Vec(0, 0, -1050), Vec(0, 0, 0), Vec(1, 1, 1),  SPEC));
    cam.AddToScene(Sphere(1000., Vec(0, 0, 1050), Vec(0, 0, 0), Vec(1, 1, 1),  SPEC));

    // cam.AddToScene(Sphere(600, Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12), Vec(), DIFF));

    //     // scene spheres
    //     cam.AddToScene(Sphere(1., Vec(0, 0, 2), Vec(), Vec(0.75, 0.25, 0.25), DIFF));
    // cam.AddToScene(Sphere(1., Vec(0, -2, 2), Vec(), Vec(0.25, 0.25, 0.75), SPEC));

    // //background
    // cam.AddToScene(Sphere(20, Vec(0, 0, 50), Vec(), Vec(1, 1, 1), DIFF));

    Vec image[w * h];

    Image im = Image(w, h);
    Ray r;

#pragma omp parallel for schedule(dynamic, 1) private(r)
    // image rows
    for (int i = 0; i < h; i++)
    {
        // image cols
        for (int j = 0; j < w; j++)
        {
            Vec c;
            // sub pixel rows
            for (int sy = 0; sy < sPixSize; sy++)
            {
                // sub pixel cols
                for (int sx = 0; sx < sPixSize; sx++)
                {
                    // sampling the pixel
                    for (int s = 0; s < numSamps; s++)
                    {
                        r = cam.GenerateRay(i, j, sx, sy);
                        c = c + cam.PixelColour(r, 0);
                    }
                }
            }
            image[i * h + j] = c / ((double)numSamps * sPixSize * sPixSize);
        }
    }

    im.Set(image);
    // Ray test = Ray(Vec(0, 0, 0), Vec(0, 0, 1));
    // HitRecord rec = cam.scene.ClosestIntersection(test, 1e-5);
    // cout << rec << endl;

    return 0;
}
    // cornell box scene
    // cam.AddToScene(Sphere(1e5, Vec(1e5 + 1, 40.8, 81.6), Vec(), Vec(.75, .25, .25), DIFF)); //Left
    // cam.AddToScene(Sphere(1e5, Vec(-1e5 + 99, 40.8, 81.6), Vec(), Vec(.25, .25, .75), DIFF));
    // cam.AddToScene(Sphere(1e5, Vec(50, 40.8, 1e5), Vec(), Vec(.75, .75, .75), DIFF));
    // cam.AddToScene(Sphere(1e5, Vec(50, 40.8, -1e5 + 170), Vec(), Vec(), DIFF));
    // cam.AddToScene(Sphere(1e5, Vec(50, 1e5, 81.6), Vec(), Vec(.75, .75, .75), DIFF));
    // cam.AddToScene(Sphere(1e5, Vec(50, -1e5 + 81.6, 81.6), Vec(), Vec(.75, .75, .75), DIFF));

    // cam.AddToScene(Sphere(16.5, Vec(27, 16.5, 47), Vec(), Vec(1, 1, 1) * .999, SPEC));
    // cam.AddToScene(Sphere(16.5, Vec(73, 16.5, 78), Vec(), Vec(1, 1, 1) * .999, SPEC));