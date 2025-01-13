// examples/ex1.cpp
//
// This example demonstrates how to use the MSz API to:
// 1. Read binary 2D/3D data from files.
// 2. Compute and count topological distortions (false minima, false maxima, and false labels)
//    in a decompressed dataset compared to the original dataset.
//
// The dataset used in this example is a 100x100 grid stored as binary files.
// The program uses MSz_count_faults to detect and report errors in the decompressed data.

#include <iostream>
#include "api/MSz.h"  // Include the MSz API header
#include <vector>
#include <fstream>

/**
 * @brief Reads binary data from a file and stores it in a vector of doubles.
 *
 * This function reads a specified number of `double` elements from a binary file.
 * It performs error checking to ensure that the file exists, is readable, and
 * contains the expected amount of data.
 *
 * @param file_path Path to the binary file.
 * @param data Vector to store the read data.
 * @param expected_num_elements Expected number of double elements to read.
 * @return true if the file is successfully read, false otherwise.
 */
bool read_binary_file(const std::string& file_path, std::vector<double>& data, size_t expected_num_elements) {
    std::ifstream file(file_path, std::ios::binary);  // Open the file in binary mode
    if (!file) {
        std::cerr << "Error: Cannot open file " << file_path << std::endl;
        return false;
    }

    // Check if the file size matches the expected number of elements
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (file_size != expected_num_elements * sizeof(double)) {
        std::cerr << "Error: File size mismatch. Expected " << expected_num_elements * sizeof(double)
                  << " bytes, but got " << file_size << " bytes." << std::endl;
        return false;
    }

    // Read the binary data into the vector
    data.resize(expected_num_elements);
    file.read(reinterpret_cast<char*>(data.data()), expected_num_elements * sizeof(double));
    if (file.gcount() != expected_num_elements * sizeof(double)) {
        std::cerr << "Error: Failed to read complete data. Only read " << file.gcount() << " bytes." << std::endl;
        return false;
    }

    file.close();
    return true;
}

/**
 * @brief Main function demonstrating the use of MSz API to count topological distortions.
 *
 * The function reads two datasets (original and decompressed), then calls
 * `MSz_count_faults` to count the number of false minima, false maxima, and
 * mislabeled points in the decompressed dataset compared to the original dataset.
 *
 * @return 0 if successful, -1 if an error occurs during file reading.
 */
int main() {
    // Variables to store the data
    std::vector<double> original_data, decompressed_data;
    int width = 100, height = 100, depth = 1;  // Dimensions of the 2D grid (set depth to 1 for 2D)
    int num_elements = width * height * depth; // Total number of elements in the grid

    // Read the original dataset from a binary file
    if (!read_binary_file("../examples/datasets/grid100x100.bin", original_data, num_elements)) {
        return -1;  // Exit if file reading fails
    }

    // Read the decompressed dataset from a binary file
    if (!read_binary_file("../examples/datasets/decp_grid100x100_sz3_rel_1e-4.bin", decompressed_data, num_elements)) {
        return -1;  // Exit if file reading fails
    }

    // Variables to store the counts of false minima, maxima, and mislabeled points
    int num_false_min = 0, num_false_max = 0, num_false_labels = 0;

    // Call the MSz API to count topological distortions
    int status = MSz_count_faults(
        original_data.data(),          // Pointer to the original data
        decompressed_data.data(),      // Pointer to the decompressed data
        num_false_min, num_false_max,  // Output: number of false minima and maxima
        num_false_labels,              // Output: number of false labels
        0,                             // Connectivity type (0 for piecewise linear)
        width, height, depth,          // Dimensions of the grid
        MSZ_ACCELERATOR_NONE           // Use pure CPU execution
    );

    // Check the status returned by the API
    if (status != MSZ_ERR_NO_ERROR) {
        std::cerr << "Error: MSz_count_faults failed with error code " << status << std::endl;
        return 0;
    }

    // Output the results
    std::cout << "Number of false minima: " << num_false_min << std::endl;
    std::cout << "Number of false maxima: " << num_false_max << std::endl;
    std::cout << "Number of false labels: " << num_false_labels << std::endl;

    return 0;
}
