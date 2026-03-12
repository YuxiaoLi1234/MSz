#include <iostream>
#include "api/MSz.h"
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

bool read_binary_file(const std::string& file_path, std::vector<double>& data, size_t expected_num_elements) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << file_path << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (file_size != expected_num_elements * sizeof(double)) {
        std::cerr << "Error: File size mismatch. Expected " << expected_num_elements * sizeof(double)
                  << " bytes, but got " << file_size << " bytes." << std::endl;
        return false;
    }

    data.resize(expected_num_elements);
    file.read(reinterpret_cast<char*>(data.data()), expected_num_elements * sizeof(double));
    file.close();
    return true;
}

int main() {
    std::vector<double> original_data, decompressed_data;
    int width = 100, height = 100, depth = 1;
    int num_elements = width * height * depth;

    // Try to load datasets, if not found, generate dummy data for testing
    if (!read_binary_file("../examples/datasets/grid100x100.bin", original_data, num_elements) ||
        !read_binary_file("../examples/datasets/decp_grid100x100_sz3_rel_1e-4.bin", decompressed_data, num_elements)) {
        std::cout << "Datasets not found, using dummy data for testing." << std::endl;
        original_data.resize(num_elements);
        decompressed_data.resize(num_elements);
        for (int i = 0; i < num_elements; ++i) {
            original_data[i] = std::sin(i * 0.1);
            decompressed_data[i] = original_data[i] + (rand() % 100) * 0.0001;
        }
    }

    int num_edits = 0;
    MSz_edit_t* edits = nullptr;

    std::cout << "Testing MSz_derive_edits with OpenMP..." << std::endl;
    int status = MSz_derive_edits(
        original_data.data(),
        decompressed_data.data(),
        nullptr,
        num_edits, &edits,
        MSZ_PRESERVE_MIN | MSZ_PRESERVE_MAX,
        0,
        width, height, depth,
        1e-4,
        MSZ_ACCELERATOR_OMP,
        0,
        4 // 4 threads
    );

    if (status != MSZ_ERR_NO_ERROR) {
        std::cerr << "Error: Failed to derive edits. Error code: " << status << std::endl;
        if (status == MSZ_ERR_NOT_IMPLEMENTED) {
            std::cout << "OpenMP not enabled in this build." << std::endl;
        } else {
            return 1;
        }
    } else {
        std::cout << "Derived " << num_edits << " edits using OpenMP." << std::endl;
    }

    if (num_edits > 0 && edits != nullptr) {
        std::cout << "\nTesting Zstd compression..." << std::endl;
        char* compressed_buffer = nullptr;
        size_t compressed_size = 0;
        int comp_status = MSz_compress_edits_zstd(num_edits, edits, &compressed_buffer, compressed_size);

        if (comp_status == MSZ_ERR_NO_ERROR) {
            std::cout << "Zstd compression successful. Compressed size: " << compressed_size << " bytes." << std::endl;

            int decomp_num_edits = 0;
            MSz_edit_t* decomp_edits = nullptr;
            int decomp_status = MSz_decompress_edits_zstd(compressed_buffer, compressed_size, decomp_num_edits, &decomp_edits);

            if (decomp_status == MSZ_ERR_NO_ERROR) {
                std::cout << "Zstd decompression successful. Decompressed " << decomp_num_edits << " edits." << std::endl;
                if (decomp_num_edits != num_edits) {
                    std::cerr << "Error: Edit count mismatch after decompression!" << std::endl;
                    return 1;
                }
                free(decomp_edits);
            } else {
                std::cerr << "Error: Zstd decompression failed. Error code: " << decomp_status << std::endl;
                return 1;
            }
            free(compressed_buffer);
        } else if (comp_status == MSZ_ERR_NOT_IMPLEMENTED) {
            std::cout << "Zstd not enabled in this build." << std::endl;
        } else {
            std::cerr << "Error: Zstd compression failed. Error code: " << comp_status << std::endl;
            return 1;
        }
    }

    if (edits) {
        free(edits);
        edits = nullptr;
        num_edits = 0;
    }

    std::cout << "\nTesting MSz_derive_edits with CUDA..." << std::endl;
    status = MSz_derive_edits(
        original_data.data(),
        decompressed_data.data(),
        nullptr,
        num_edits, &edits,
        MSZ_PRESERVE_MIN | MSZ_PRESERVE_MAX,
        0,
        width, height, depth,
        1e-4,
        MSZ_ACCELERATOR_CUDA,
        0,
        0
    );

    if (status != MSZ_ERR_NO_ERROR) {
        if (status == MSZ_ERR_NOT_IMPLEMENTED || status == MSZ_ERR_NO_AVAILABLE_GPU) {
            std::cout << "CUDA not available or not enabled in this build." << std::endl;
        } else {
            std::cerr << "Error: Failed to derive edits with CUDA. Error code: " << status << std::endl;
            return 1;
        }
    } else {
        std::cout << "Derived " << num_edits << " edits using CUDA." << std::endl;
    }

    if (edits) free(edits);
    std::cout << "\nTest completed successfully." << std::endl;
    return 0;
}
