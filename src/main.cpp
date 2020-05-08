#include <iostream>
#include <cmath>
#include "vector.h"
#include "sphere.h"
#include "plane.h"
#include "image.h"
#include "hitrecord.h"
#include "camera.h"
#include "ptmath.h"

using namespace std;

// RUN: clear && make && ./bin/main
int main()
{

    int w = 500;
    int h = 500;
    int numSamps = 1000;
    int sPixSize = 2;

    double boxSize = 10;

    Camera cam = Camera(2 * boxSize, w, h, Vec(0, 0., -boxSize + 0.1), Vec(0, 0, 1.), Vec(0, 1, 0), PTMath::PI / 3.0);

    Plane front = Plane(Vec(0, 0, -1), Vec(0, 0, boxSize), Vec(), Vec(1, 1, 1), DIFF);
    Plane right = Plane(Vec(-1, 0, 0), Vec(boxSize, 0, 0), Vec(), Vec(0, 1, 0), DIFF);
    Plane back = Plane(Vec(0, 0, 1), Vec(0, 0, -boxSize), Vec(), Vec(), DIFF);
    Plane left = Plane(Vec(1, 0, 0), Vec(-boxSize, 0, 0), Vec(), Vec(1, 0, 0), DIFF);
    Plane top = Plane(Vec(0, -1, 0), Vec(0, boxSize, 0), Vec(1, 1, 1), Vec(), DIFF);
    Plane bottom = Plane(Vec(0, 1, 0), Vec(0, -boxSize, 0), Vec(), Vec(1, 1, 1), DIFF); // and the light

    Sphere glass = Sphere(4, Vec(5, -6, 0), Vec(0, 0, 0), Vec(1, 1, 1), REFR);
    Sphere mirr = Sphere(4, Vec(-5, -6, 3), Vec(0, 0, 0), Vec(1, 1, 1), SPEC);

    cam.AddToScene(&front);
    cam.AddToScene(&right);
    cam.AddToScene(&back);
    cam.AddToScene(&left);
    cam.AddToScene(&top);
    cam.AddToScene(&bottom);

    cam.AddToScene(&glass);
    cam.AddToScene(&mirr);


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

    return 0;
}
