#include <cmath>
#include "image.h"

Image::Image(int _dimX, int _dimY, int index)
{
    dimX = _dimX;
    dimY = _dimY;
    const char* str = "result";
    const char* ext = ".ppm";
    char fileName[sizeof(str) + sizeof(ext) + 1];
    snprintf(fileName, sizeof(fileName), "%s %d%s", str, index, ext);
    file = fopen(fileName, "wb");
    fprintf(file, "P6\n%d %d\n255\n", dimX, dimY);
}

Image::~Image()
{
    fclose(file);
}

void Image::Set(Vec *colour)
{

    for (int i = 0; i < dimY * dimX; i++)
    {
        Vec c = colour[i];
        unsigned char col[3];
        double r = pow(c.x, 1.0 / 2) * 255;
        double g = pow(c.y, 1.0 / 2) * 255;
        double b = pow(c.z, 1.0 / 2) * 255;
        col[0] = r > 255 ? 255 : r < 0 ? 0 : r;
        col[1] = g > 255 ? 255 : g < 0 ? 0 : g;
        col[2] = b > 255 ? 255 : b < 0 ? 0 : b;
        fwrite(col, 1, 3, file);
    }
}

void Image::Close()
{
    fclose(file);
}