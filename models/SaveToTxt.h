#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include "../matrix.h" // Ensure this file defines or includes the definition of Matrix
#include <string>
// Removed redundant include as "../matrix.h" already includes the required file

void save_to_txt(const std::string& filename, const std::vector<Matrix>& data);

std::vector<Matrix> read_from_txt(const std::string& filename);

void save_to_bin(const std::string& filename, const std::vector<Matrix>& data);

std::vector<Matrix> read_from_bin(const std::string& filename);