#pragma once
#include <fstream>
#include "vector.h"

struct Image
{   
    int dimX, dimY; // number of pixels in the x and y directions respectively
    FILE *file;
    Image(int _dimX, int _dimY, int index);
    ~Image();
    void Set(Vec* colour); // prints the value of colour into the next entry inthe file
    void Close(); // closes the file -- obsolete since that is now handled in the destructor
};