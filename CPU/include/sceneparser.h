#pragma once
#include <string>
#include <vector>
#include "solid.h"

namespace SceneParser
{
    
    std::vector<std::string> ReadFile(std::string fName);
    std::vector<Solid> Parse(std::vector<std::string>);
    
}