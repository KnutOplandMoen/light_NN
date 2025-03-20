#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include "../matrix.h"
#include <string>

void save_to_txt(const std::string& filename, const std::vector<Matrix>& data) {
    std::string file_n = "c:\\Users\\knuto\\Documents\\programering\\NN\\light_NN\\models\\" + filename;
    std::ofstream file(file_n);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file: " << filename << std::endl;
        throw std::invalid_argument("Could not open the file: " + filename);
    }

    std::cout << "File opened successfully: " << filename << std::endl;

    file << "Input data:\n"; // This should be written to the file
    for (int i = 0; i < data.size(); ++i) {
        file << data[i] << std::endl;
    }

    if (file.fail()) {
        std::cerr << "Error: Failed to write to the file." << std::endl;
    }

    // Close the file to flush the output
    file.close();

    std::cout << "Data successfully saved to " << filename << std::endl;
}