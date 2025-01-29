#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include "api/MSz.h" 

void print_usage() {
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  ./MSz_CLI count_faults <original_file> <decompressed_file> <width> <height> <depth> <connectivity> --mode <none|omp|cuda>" << std::endl;
    std::cerr << "  ./MSz_CLI derive_edits <original_file> <decompressed_file> <width> <height> <depth> <connectivity> <error_bound> <preserve_min> <preserve_max> <preserve_path> --mode <none|omp|cuda>" << std::endl;
    std::cerr << "  ./MSz_CLI apply_edits <decompressed_file> <edits_file> <width> <height> <depth> --mode <none|omp|cuda>" << std::endl;
}

void save_edits(const std::string& file_path, const MSz_edit_t* edits, int num_edits) {
    if (num_edits <= 0 || edits == nullptr) {
        std::cerr << "Error: Invalid edits data to save." << std::endl;
        exit(MSZ_ERR_INVALID_INPUT);
    }

    std::ofstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << file_path << " for writing." << std::endl;
        exit(MSZ_ERR_UNKNOWN_ERROR);
    }

    file.write(reinterpret_cast<const char*>(&num_edits), sizeof(int));
    file.write(reinterpret_cast<const char*>(edits), num_edits * sizeof(MSz_edit_t));

    if (!file) {
        std::cerr << "Error: Failed to write edits to file " << file_path << std::endl;
        exit(MSZ_ERR_UNKNOWN_ERROR);
    }

    file.close();
}


void load_edits(const std::string& file_path, std::vector<MSz_edit_t>& edits) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open edits file " << file_path << std::endl;
        exit(MSZ_ERR_INVALID_INPUT);
    }

    int num_edits = 0;
    file.read(reinterpret_cast<char*>(&num_edits), sizeof(int));

    if (num_edits <= 0) {
        std::cerr << "Error: Invalid number of edits in file " << file_path << std::endl;
        exit(MSZ_ERR_INVALID_INPUT);
    }

    edits.resize(num_edits);
    file.read(reinterpret_cast<char*>(edits.data()), num_edits * sizeof(MSz_edit_t));

    if (!file) {
        std::cerr << "Error: Failed to read edits from file " << file_path << std::endl;
        exit(MSZ_ERR_UNKNOWN_ERROR);
    }

    file.close();
}


void load_data(const std::string& file_path, std::vector<double>& data) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << file_path << std::endl;
        exit(MSZ_ERR_INVALID_INPUT);
    }
    

    file.seekg(0, std::ios::end);
    size_t num_elements = file.tellg() / sizeof(double);
    file.seekg(0, std::ios::beg);
    
    data.resize(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(double));
    file.close();
}

void save_data(const std::string& file_path, const std::vector<double>& data) {
    std::ofstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot write to file " << file_path << std::endl;
        exit(MSZ_ERR_UNKNOWN_ERROR);
    }
    
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
    file.close();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {  
        std::cerr << "Error: No command provided." << std::endl;
        print_usage();
        return MSZ_ERR_INVALID_INPUT;
    }

    std::string command = argv[1];

    if (command == "count_faults") {
        if (argc != 9) {  
            std::cerr << "Error: Incorrect number of arguments for count_faults." << std::endl;
            print_usage();
            return MSZ_ERR_INVALID_INPUT;
        }

        std::string original_file = argv[2];
        std::string decompressed_file = argv[3];
        int width = std::stoi(argv[4]);
        int height = std::stoi(argv[5]);
        int depth = std::stoi(argv[6]);
        int connectivity = std::stoi(argv[7]);
        std::string mode = argv[8];

        int accelerator = (mode == "omp") ? MSZ_ACCELERATOR_OMP :
                          (mode == "cuda") ? MSZ_ACCELERATOR_CUDA : MSZ_ACCELERATOR_NONE;

        std::cout << "Using " << (mode == "omp" ? "OpenMP" : mode == "cuda" ? "CUDA" : "pure CPU") << " execution." << std::endl;

        std::vector<double> original_data, decompressed_data;
        load_data(original_file, original_data);
        load_data(decompressed_file, decompressed_data);

        int num_false_min = 0, num_false_max = 0, num_false_labels = 0;
        int status = MSz_count_faults(
            original_data.data(), decompressed_data.data(),
            num_false_min, num_false_max, num_false_labels,
            connectivity, width, height, depth, accelerator
        );

        if (status == MSZ_ERR_NO_ERROR) {
            std::cout << "Count faults succeeded." << std::endl;
            std::cout << "False minima: " << num_false_min << std::endl;
            std::cout << "False maxima: " << num_false_max << std::endl;
            std::cout << "False labels: " << num_false_labels << std::endl;
        } else {
            std::cerr << "Error in count_faults. Error code: " << status << std::endl;
            return status;
        }

    } else if (command == "derive_edits") {
        if (argc != 13) {
            std::cout<<argc<<std::endl;
            std::cerr << "Error: Incorrect number of arguments for derive_edits." << std::endl;
            print_usage();
            return MSZ_ERR_INVALID_INPUT;
        }

        std::string original_file = argv[2];
        std::string decompressed_file = argv[3];
        int width = std::stoi(argv[4]);
        int height = std::stoi(argv[5]);
        int depth = std::stoi(argv[6]);
        int connectivity = std::stoi(argv[7]);
        double error_bound = std::stod(argv[8]);
        int preserve_min = std::stoi(argv[9]);
        int preserve_max = std::stoi(argv[10]);
        int preserve_path = std::stoi(argv[11]);
        std::string mode = argv[12];


        int accelerator = (mode == "omp") ? MSZ_ACCELERATOR_OMP :
                          (mode == "cuda") ? MSZ_ACCELERATOR_CUDA : MSZ_ACCELERATOR_NONE;

        std::cout << "Using " << (mode == "omp" ? "OpenMP" : mode == "cuda" ? "CUDA" : "pure CPU") << " execution." << std::endl;

        unsigned int preservation_options = 0;
        if (preserve_min) preservation_options |= MSZ_PRESERVE_MIN;
        if (preserve_max) preservation_options |= MSZ_PRESERVE_MAX;
        if (preserve_path) preservation_options |= MSZ_PRESERVE_PATH;

        std::vector<double> original_data, decompressed_data;
        load_data(original_file, original_data);
        load_data(decompressed_file, decompressed_data);

        int num_edits = 0;
        MSz_edit_t* edits = nullptr;

        int status = MSz_derive_edits(
            original_data.data(), decompressed_data.data(),
            nullptr, num_edits, &edits,
            preservation_options, connectivity, width, height, depth, error_bound, accelerator
        );

        if (status == MSZ_ERR_NO_ERROR) {
            std::cout << "Derive edits succeeded. Number of edits: " << num_edits << std::endl;
            save_edits("edits.bin", edits, num_edits);
            std::cout << "Edits saved to edits.bin" << std::endl;
            free(edits);
        } else {
            std::cerr << "Error in derive_edits. Error code: " << status << std::endl;
            return status;
        }

    } else if (command == "apply_edits") {
        if (argc != 9) {
            std::cerr << "Error: Incorrect number of arguments for apply_edits." << std::endl;
            print_usage();
            return MSZ_ERR_INVALID_INPUT;
        }

        std::string decompressed_file = argv[2];
        std::string edits_file = argv[3];
        int width = std::stoi(argv[4]);
        int height = std::stoi(argv[5]);
        int depth = std::stoi(argv[6]);
        std::string mode = argv[7];

        int accelerator = (mode == "omp") ? MSZ_ACCELERATOR_OMP :
                          (mode == "cuda") ? MSZ_ACCELERATOR_CUDA : MSZ_ACCELERATOR_NONE;

        std::cout << "Using " << (mode == "omp" ? "OpenMP" : mode == "cuda" ? "CUDA" : "pure CPU") << " execution." << std::endl;

        std::vector<double> decompressed_data;
        load_data(decompressed_file, decompressed_data);

        std::vector<MSz_edit_t> edits;
        load_edits(edits_file, edits);

        int status = MSz_apply_edits(
            decompressed_data.data(), edits.size(), edits.data(),
            width, height, depth, accelerator
        );

        if (status == MSZ_ERR_NO_ERROR) {
            std::cout << "Apply edits succeeded." << std::endl;
            save_data("edited_data.bin", decompressed_data);
            std::cout << "Edited data saved to edited_data.bin" << std::endl;
        } else {
            std::cerr << "Error in apply_edits. Error code: " << status << std::endl;
            return status;
        }

    } else {
        std::cerr << "Error: Unknown command '" << command << "'." << std::endl;
        print_usage();
        return MSZ_ERR_INVALID_INPUT;
    }

    return MSZ_ERR_NO_ERROR;
}
