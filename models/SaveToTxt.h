#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include "../matrix.h"
#include <string>

void save_to_txt(const std::string& filename, const std::vector<Matrix>& data);