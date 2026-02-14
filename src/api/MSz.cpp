#include "api/MSz.h"
#include <fstream>
#include <cstdint>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <unordered_map>
#include <random>
#include <atomic>
#include <string>
#include <iostream>
#include <unordered_set>
#include <set>
#include <map>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <utility>
#include <iomanip>
#include <chrono>
#include <random>
#include <cstdio>

#ifdef MSZ_ENABLE_ZSTD
    #include <zstd.h>
#endif

#ifdef MSZ_ENABLE_OPENMP
    #include <omp.h>
    #include "MSz_omp.h"
#endif

#include "MSz_serial.h"
#include "MSz_globals.h"

#ifdef MSZ_ENABLE_CUDA
    #include "device_launch_parameters.h"
    #include "cuda_runtime.h"
    #include "cublas_v2.h"
    #include "MSz_CUDA.h"
#endif



int MSz_derive_edits(
        const double *original_data,   // Input: original data array
        const double *decompressed_data, // Input: decompressed data array
        double *edited_decompressed_data, // Output: edited data array (optional, can be nullptr)
        int &num_edits,                // Output: number of edits
        MSz_edit_t **edits,
        unsigned int preservation_options, //bitset for preservation options
        unsigned int connectivity_type,    // connectivity type specifier
        int W, int H, int D,           // Dimensions of the data
        double rel_err_bound,          // Relative error bound for edits
        int accelerator,  // hardware accelerator
        int device_id,
        int num_omp_threads) {

    if (!original_data || !decompressed_data || W <= 0 || H <= 0 || D <= 0) {
        printf("file failed\n");
        return MSZ_ERR_INVALID_INPUT;
    }

    if(connectivity_type != 0 && connectivity_type != 1) {
        printf("MSZ_ERR_INVALID_CONNECTIVITY_TYPE failed\n");
        return MSZ_ERR_INVALID_CONNECTIVITY_TYPE;
    }

    std::vector<double> input_data(original_data, original_data+W*H*D);
    std::vector<double> decp_data_(decompressed_data, decompressed_data+W*H*D);
    std::vector<double> decp_data_copy(decompressed_data, decompressed_data+W*H*D);

    
    int preserve_min = 0; // Flag for preserving minima
    int preserve_max = 0; // Flag for preserving maxima
    int preserve_path = 0; // Flag for preserving separatrices

    // Check each option using bitwise AND (&) and set the corresponding flag
    if (preservation_options & MSZ_PRESERVE_MIN) {
        preserve_min = 1; // Enable preserving minima
    }
    if (preservation_options & MSZ_PRESERVE_MAX) {
        preserve_max = 1; // Enable preserving maxima
    }
    if (preservation_options & MSZ_PRESERVE_PATH) {
        preserve_path = 1; // Enable preserving separatrices
    }

    int data_size = input_data.size();

    auto min_it = std::min_element(input_data.begin(), input_data.end());
    auto max_it = std::max_element(input_data.begin(), input_data.end());
    double minValue = *min_it;
    double maxValue = *max_it;
    
    double bound = (maxValue-minValue)*rel_err_bound;

    std::vector<int> or_direction_as, or_direction_ds, de_direction_as1, de_direction_ds1, dec_label1, or_label1;
    or_direction_as.resize(data_size);
    or_direction_ds.resize(data_size);
    de_direction_as1.resize(data_size);
    de_direction_ds1.resize(data_size);
    or_label1.resize(data_size*2, -1);
    dec_label1.resize(data_size*2, -1);
    

    std::vector<int>* dev_a = &or_direction_as;
    std::vector<int>* dev_b = &or_direction_ds;
    std::vector<int>* dev_c = &de_direction_as1;
    std::vector<int>* dev_d = &de_direction_ds1;
    std::vector<double>* dev_e = &input_data;
    std::vector<double>* dev_f = &decp_data_copy;
    
    
    
    std::vector<int>* dev_q = &dec_label1;
    std::vector<int>* dev_m = &or_label1;
    
    
    int status = MSZ_ERR_NO_ERROR;

    if (accelerator == MSZ_ACCELERATOR_CUDA) {
        #ifdef MSZ_ENABLE_CUDA
            status = fix_process(dev_a, dev_b, dev_c, dev_d, dev_e, dev_f, dev_m, dev_q,
                                W, H, D, bound, preserve_min, preserve_max, preserve_path, connectivity_type,
                                device_id);
        #else
            return MSZ_ERR_NOT_IMPLEMENTED;
        #endif
    } 
    else if (accelerator == MSZ_ACCELERATOR_OMP) {
        #ifdef MSZ_ENABLE_OPENMP
                if (num_omp_threads <= 0) {
                    return MSZ_ERR_INVALID_THREAD_COUNT;
                }
                omp_set_num_threads(num_omp_threads);
                status = fix_process_omp(dev_a, dev_b, dev_c, dev_d, dev_e, dev_f, dev_q, dev_m,
                                        W, H, D, bound, preserve_min, preserve_max, preserve_path, connectivity_type);
        #else
                return MSZ_ERR_NOT_IMPLEMENTED;
        #endif
    } 
    else if (accelerator == MSZ_ACCELERATOR_NONE) {
        status = fix_process_cpu(dev_a, dev_b, dev_c, dev_d, dev_e, dev_f, dev_q, dev_m,
                                W, H, D, bound, preserve_min, preserve_max, preserve_path, connectivity_type);
    } 
    else {
        return MSZ_ERR_NOT_IMPLEMENTED;
    }

    if(status != MSZ_ERR_NO_ERROR) return status;
        


    std::vector<uint32_t> indexs;
    std::vector<double> deltas;

    
    for (uint32_t i=0;i<input_data.size();i++){
        
        if (decp_data_copy[i]!=decompressed_data[i]){
            indexs.push_back(i);
            deltas.push_back(decp_data_copy[i] - decompressed_data[i]);
        }
    }

    num_edits = indexs.size();
    
    if (num_edits == 0) {
        *edits = nullptr;
        return MSZ_ERR_NO_ERROR;
    }

    // Allocate memory for edits
    *edits = (MSz_edit_t *)malloc(num_edits * sizeof(MSz_edit_t));
    
    if (*edits == nullptr) {
        std::cerr << "Memory allocation failed for edits." << std::endl;
        num_edits = 0;
        return MSZ_ERR_OUT_OF_MEMORY;
    }

    // Populate the edits array
    
    for (int i = 0; i < num_edits; i++) {
        (*edits)[i].index = indexs[i];
        (*edits)[i].offset = deltas[i];
    }
    

    // Optional: Apply edits to the decompressed data
    if (edited_decompressed_data) {
        std::copy(decp_data_copy.begin(), decp_data_copy.end(), edited_decompressed_data);
    }

    return MSZ_ERR_NO_ERROR;
}

int MSz_count_faults(
        const double *original_data, // Input: original data array
        const double *decompressed_data, // Input: decompressed data array
        int &num_false_min, // Output: number of false minimum
        int &num_false_max, // Output: number of false maximum
        int &num_false_labels, // Output: number of data points with wrong Morse-smale segmentation labels
        unsigned int connectivity_type, // connectivity type specifier
        int W, int H, int D, // Dimensions of the data
        int accelerator,  // hardware accelerator
        int device_id,
        int num_omp_threads  
    ){
    if (!original_data || !decompressed_data || W <= 0 || H <= 0 || D <= 0) {
        return MSZ_ERR_INVALID_INPUT;
    }
    if(connectivity_type != 0 && connectivity_type != 1) return MSZ_ERR_INVALID_CONNECTIVITY_TYPE;
    std::vector<double> input_data(original_data, original_data+W*H*D);
    std::vector<double> decp_data_(decompressed_data, decompressed_data+W*H*D);
    
    int data_size = W*H*D;

    std::vector<int> or_direction_as, or_direction_ds, de_direction_as1, de_direction_ds1, dec_label1, or_label1;
    or_direction_as.resize(data_size);
    or_direction_ds.resize(data_size);
    de_direction_as1.resize(data_size);
    de_direction_ds1.resize(data_size);
    or_label1.resize(data_size*2, -1);
    dec_label1.resize(data_size*2, -1);
    

    std::vector<int>* dev_a = &or_direction_as;
    std::vector<int>* dev_b = &or_direction_ds;
    std::vector<int>* dev_c = &de_direction_as1;
    std::vector<int>* dev_d = &de_direction_ds1;
    std::vector<double>* dev_e = &input_data;
    std::vector<double>* dev_f = &decp_data_;

    std::vector<int>* dev_q = &dec_label1;
    std::vector<int>* dev_m = &or_label1;
    
    int status = MSZ_ERR_NOT_IMPLEMENTED;

    if (accelerator == MSZ_ACCELERATOR_CUDA) {
        #ifdef MSZ_ENABLE_CUDA
                status = count_false_cases(
                    dev_a, dev_b, dev_c, dev_d, dev_e, dev_f, dev_m, dev_q,
                    W, H, D, connectivity_type,
                    num_false_min, num_false_max, num_false_labels,
                    device_id
                );
        #else
                return MSZ_ERR_NOT_IMPLEMENTED;
        #endif
    } else if (accelerator == MSZ_ACCELERATOR_OMP) {
        #ifdef MSZ_ENABLE_OPENMP
                if (num_omp_threads <= 0) {
                    return MSZ_ERR_INVALID_THREAD_COUNT;
                }
                omp_set_num_threads(num_omp_threads);
                status = count_false_cases_omp(
                    dev_a, dev_b, dev_c, dev_d, dev_e, dev_f, dev_m, dev_q,
                    W, H, D, connectivity_type,
                    num_false_min, num_false_max, num_false_labels
                );
        #else
                return MSZ_ERR_NOT_IMPLEMENTED;
        #endif
    } else if (accelerator == MSZ_ACCELERATOR_NONE) {
        status = count_false_cases_cpu(
            dev_a, dev_b, dev_c, dev_d, dev_e, dev_f, dev_m, dev_q,
            W, H, D, connectivity_type,
            num_false_min, num_false_max, num_false_labels
        );
    }
    return status;
}

double MSz_calculate_false_label_ratio(
        int num_false_labels,
        int W, int H, int D){
    if(W <= 0 || H <= 0 || D <= 0 || num_false_labels < 0) return  MSZ_ERR_INVALID_INPUT;
    return static_cast<double>(num_false_labels) / (W * H * D);
}

int MSz_compress_edits_zstd(
    int num_edits,
    MSz_edit_t *edits,
    char **compressed_buffer,
    size_t &compressed_size) {

    #ifndef MSZ_ENABLE_ZSTD
        return MSZ_ERR_NOT_IMPLEMENTED; // Zstd disabled
    #else

        if (num_edits < 0 || edits == nullptr || compressed_buffer == nullptr) {
            return MSZ_ERR_INVALID_INPUT;
        }

        // Step 1: Allocate temporary arrays for index and offset
        uint32_t *index_array = (uint32_t *)malloc(num_edits * sizeof(uint32_t));
        double *offset_array = (double *)malloc(num_edits * sizeof(double));
        if (!index_array || !offset_array) {
            free(index_array);
            free(offset_array);
            return MSZ_ERR_OUT_OF_MEMORY;
        }

        // Step 2: Extract index and offset from edits
        for (int i = 0; i < num_edits; ++i) {
            index_array[i] = edits[i].index;
            offset_array[i] = edits[i].offset;
        }

        // Step 3: Convert index_array to diffs
        std::vector<uint32_t> diffs;
        if (num_edits > 0) {
            diffs.push_back(index_array[0]); // First index remains unchanged
            for (int i = 1; i < num_edits; ++i) {
                diffs.push_back(index_array[i] - index_array[i - 1]); // Store differences
            }
        }

        free(index_array); // No longer need the original index array

        // Step 4: Allocate combined buffer
        size_t combined_buffer_size = diffs.size() * sizeof(uint32_t) + num_edits * sizeof(double);
        char *combined_buffer_temp = (char *)malloc(combined_buffer_size);
        if (combined_buffer_temp == nullptr) {
            free(offset_array);
            return MSZ_ERR_OUT_OF_MEMORY;
        }

        // Step 5: Copy diffs and offsets into the combined buffer
        char *ptr = combined_buffer_temp;
        memcpy(ptr, diffs.data(), diffs.size() * sizeof(uint32_t));
        ptr += diffs.size() * sizeof(uint32_t);
        memcpy(ptr, offset_array, num_edits * sizeof(double));

        free(offset_array); // No longer need the offset array

        // Step 6: Compress the combined buffer
        size_t max_compressed_size = ZSTD_compressBound(combined_buffer_size);
        *compressed_buffer = (char *)malloc(max_compressed_size);
        if (*compressed_buffer == nullptr) {
            free(combined_buffer_temp);
            return MSZ_ERR_OUT_OF_MEMORY;
        }

        compressed_size = ZSTD_compress(
            *compressed_buffer,
            max_compressed_size,
            combined_buffer_temp,
            combined_buffer_size,
            1 // Compression level
        );

        if (ZSTD_isError(compressed_size)) {
            free(combined_buffer_temp);
            free(*compressed_buffer);
            *compressed_buffer = nullptr;
            return MSZ_ERR_EDITS_COMPRESSION_FAILED;
        }

        // Step 7: Free temporary combined buffer
        free(combined_buffer_temp);

        return MSZ_ERR_NO_ERROR; // Success
    #endif
}

int MSz_decompress_edits_zstd(
    const char *compressed_buffer,
    size_t compressed_size,
    int &num_edits,
    MSz_edit_t **edits) {

    #ifndef MSZ_ENABLE_ZSTD
        return MSZ_ERR_NOT_IMPLEMENTED; // Zstd disabled
    #else
        if (compressed_buffer == nullptr || compressed_size == 0 || edits == nullptr) {
            return MSZ_ERR_INVALID_INPUT;
        }

        // Step 1: Decompress the combined buffer
        size_t decompressed_size = ZSTD_getFrameContentSize(compressed_buffer, compressed_size);
        if (decompressed_size == ZSTD_CONTENTSIZE_ERROR || decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
            return MSZ_ERR_EDITS_DECOMPRESSION_FAILED;
        }

        char *decompressed_buffer = (char *)malloc(decompressed_size);
        if (decompressed_buffer == nullptr) {
            return MSZ_ERR_OUT_OF_MEMORY;
        }

        size_t result = ZSTD_decompress(
            decompressed_buffer,
            decompressed_size,
            compressed_buffer,
            compressed_size
        );

        if (ZSTD_isError(result)) {
            free(decompressed_buffer);
            return MSZ_ERR_EDITS_DECOMPRESSION_FAILED;
        }

        // Step 2: Extract num_edits
        size_t diff_size = decompressed_size / (sizeof(uint32_t) + sizeof(double)) * sizeof(uint32_t);
        num_edits = diff_size / sizeof(uint32_t);

        // Step 3: Allocate memory for edits
        *edits = (MSz_edit_t *)malloc(num_edits * sizeof(MSz_edit_t));
        if (*edits == nullptr) {
            free(decompressed_buffer);
            return MSZ_ERR_OUT_OF_MEMORY;
        }

        // Step 4: Parse decompressed buffer
        const char *ptr = decompressed_buffer;

        // Extract diffs
        std::vector<uint32_t> diffs(num_edits);
        memcpy(diffs.data(), ptr, num_edits * sizeof(uint32_t));
        ptr += num_edits * sizeof(uint32_t);

        // Extract offsets
        std::vector<double> offsets(num_edits);
        memcpy(offsets.data(), ptr, num_edits * sizeof(double));

        // Step 5: Reconstruct index array from diffs
        std::vector<uint32_t> index_array(num_edits);
        if (num_edits > 0) {
            index_array[0] = diffs[0]; // First index remains unchanged
            for (int i = 1; i < num_edits; ++i) {
                index_array[i] = index_array[i - 1] + diffs[i]; // Reconstruct full index array
            }
        }

        // Step 6: Populate edits
        for (int i = 0; i < num_edits; ++i) {
            (*edits)[i].index = index_array[i];
            (*edits)[i].offset = offsets[i];
        }

        // Step 7: Free temporary buffer
        free(decompressed_buffer);
        
        return MSZ_ERR_NO_ERROR; // Success
    #endif
}

int MSz_apply_edits(
    double *decompressed_data,
    int num_edits,
    const MSz_edit_t *edits,
    int W, int H, int D,
    int accelerator,
    int device_id,
    int num_omp_threads) {
    
    if (!decompressed_data || !edits || num_edits <= 0 || W <= 0 || H <= 0 || D <= 0) {
        return MSZ_ERR_INVALID_INPUT;
    }

    
    if (accelerator == MSZ_ACCELERATOR_NONE) {
        
        for (int i = 0; i < num_edits; ++i) {
            int index = edits[i].index;
            double offset = edits[i].offset;
            decompressed_data[index] += offset;
        }
    }
    else if (accelerator == MSZ_ACCELERATOR_OMP) {
        #ifdef MSZ_ENABLE_OPENMP

        #pragma omp parallel for num_threads(num_omp_threads)
        for (int i = 0; i < num_edits; ++i) {
            int index = edits[i].index;
            double offset = edits[i].offset;
            decompressed_data[index] += offset;
        }

        #else
            return MSZ_ERR_NOT_IMPLEMENTED;
        #endif
    }

    else if (accelerator == MSZ_ACCELERATOR_CUDA) {
        #ifdef MSZ_ENABLE_CUDA
            MSz_apply_edits_cuda(decompressed_data,
                                 num_edits, 
                                 edits, W, H,D, accelerator, device_id);
            return 0;
        #else
            return MSZ_ERR_NOT_IMPLEMENTED;
        #endif
    }
    else {
        return MSZ_ERR_UNKNOWN_ERROR;
    }

    
    return MSZ_ERR_NO_ERROR;
}




