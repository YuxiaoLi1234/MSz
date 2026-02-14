// examples/ex2.cpp
//
// This example demonstrates how to use the MSz API to:
// 1. Read binary data representing a 100x100 grid from files.
// 2. Derive topology-preserving edits using `MSz_derive_edits`.
// 3. Apply the derived edits to the decompressed data using `MSz_apply_edits`.
// 
// This example operates on two datasets:
// - The original dataset: "../examples/datasets/grid100x100.bin"
// - The decompressed dataset: "../examples/datasets/decp_grid100x100_sz3_rel_1e-4.bin"
// 
// The edits derived and applied ensure that the topological features (minima and maxima)
// are preserved in the decompressed dataset within a specified error bound.

#include <iostream>
#include "api/MSz.h"  // Include the MSz API header file
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
 * @brief Main function demonstrating how to derive and apply topology-preserving edits.
 *
 * This function performs the following steps:
 * 1. Reads the original and decompressed datasets from binary files.
 * 2. Derives topology-preserving edits using `MSz_derive_edits`.
 * 3. Applies the derived edits to the decompressed data using `MSz_apply_edits`.
 *
 * @return 0 if successful, -1 if an error occurs during file reading.
 */
int main() {
    // Variables to store the data
    std::vector<double> original_data, decompressed_data;
    int width = 100, height = 100, depth = 1;  // Dimensions of the grid (set depth to 1 for 2D)
    int num_elements = width * height * depth; // Total number of elements in the grid

    // Read the original dataset from a binary file
    if (!read_binary_file("../examples/datasets/grid100x100.bin", original_data, num_elements)) {
        return -1;  // Exit if file reading fails
    }

    // Read the decompressed dataset from a binary file
    if (!read_binary_file("../examples/datasets/decp_grid100x100_sz3_rel_1e-4.bin", decompressed_data, num_elements)) {
        return -1;  // Exit if file reading fails
    }

    // Variables to store the derived edits
    int num_edits = 0;
    MSz_edit_t* edits = nullptr;

    // Call MSz API to derive topology-preserving edits
    int status = MSz_derive_edits(
        original_data.data(),          // Pointer to the original data
        decompressed_data.data(),      // Pointer to the decompressed data
        nullptr,                       // Optional: edited data output (set to nullptr)
        num_edits, &edits,             // Output: number of edits and edits array
        MSZ_PRESERVE_MIN | MSZ_PRESERVE_MAX, // Preserve minima and maxima
        0,                             // Connectivity type: piecewise linear
        width, height, depth,          // Dimensions of the grid
        1e-4,                          // Relative error bound
        MSZ_ACCELERATOR_NONE           // Use CPU for computation
    );

    if (status == MSZ_ERR_NO_ERROR) {
        std::cout << "Number of edits: " << num_edits << std::endl;
    } else {
        std::cerr << "Error: Failed to derive edits. Error code: " << status << std::endl;
        free(edits);  // Free the allocated memory for edits
        return 0;
    }

    // Call MSz API to apply the derived edits to the decompressed data
    status = MSz_apply_edits(
        decompressed_data.data(),  // Pointer to the decompressed data
        num_edits, edits,          // Number of edits and edits array
        width, height, depth,      // Dimensions of the grid
        MSZ_ACCELERATOR_NONE       // Use CPU for computation
    );

    if (status == MSZ_ERR_NO_ERROR) {
        std::cerr << "Successfully applied edits!" << std::endl;
        free(edits);  // Free the allocated memory for edits
    } else {
        std::cerr << "Error: Failed to apply edits. Error code: " << status << std::endl;
    }

    return 0;
}