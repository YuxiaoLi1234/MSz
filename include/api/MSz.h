#ifndef _MSZ_API_H
#define _MSZ_API_H
#include <cstdint>
#include <stddef.h>

// Enum for preservation options using bitset representation
enum {
    MSZ_PRESERVE_MIN = 0x1,    // Preserve minima
    MSZ_PRESERVE_MAX = 0x2,    // Preserve maxima
    MSZ_PRESERVE_SADDLE = 0x4, // Preserve saddle points
    MSZ_PRESERVE_PATH = 0x8,   // Preserve separatrices that connecting minima and maxima
};

/**
 * @brief Error codes for API functions.
 *
 * These error codes indicate the status of API function calls. 
 * A value of `MSZ_ERR_NO_ERROR` indicates successful execution, while 
 * other values indicate specific errors.
 *
 * Values:
 * - `MSZ_ERR_NO_ERROR`: Operation completed successfully.
 * - `MSZ_ERR_INVALID_INPUT`: Input parameters are invalid.
 * - `MSZ_ERR_INVALID_CONNECTIVITY_TYPE`: Invalid connectivity type was specified.
 * - `MSZ_ERR_NO_AVAILABLE_GPU`: No available GPU for computation.
 * - `MSZ_ERR_OUT_OF_MEMORY`: Memory allocation failed.
 * - `MSZ_ERR_UNKNOWN_ERROR`: An unknown error occurred.
 * - `MSZ_ERR_EDITS_COMPRESSION_FAILED`: Compression of edits failed.
 * - `MSZ_ERR_EDITS_DECOMPRESSION_FAILED`: Decompression of edits failed.
 * - `MSZ_ERR_NOT_IMPLEMENTED`: The requested feature or functionality is not yet implemented.
 * - `MSZ_ERR_INVALID_THREAD_COUNT`: The specified thread count for OpenMP is invalid (e.g., less than 1).
 */
enum {
    MSZ_ERR_NO_ERROR = 0,
    MSZ_ERR_INVALID_INPUT,
    MSZ_ERR_INVALID_CONNECTIVITY_TYPE,
    MSZ_ERR_NO_AVAILABLE_GPU,
    MSZ_ERR_OUT_OF_MEMORY,
    MSZ_ERR_UNKNOWN_ERROR,
    MSZ_ERR_EDITS_COMPRESSION_FAILED,
    MSZ_ERR_EDITS_DECOMPRESSION_FAILED,
    MSZ_ERR_NOT_IMPLEMENTED,
    MSZ_ERR_INVALID_THREAD_COUNT
};


// Hardware accelerators
/**
 * @brief Supported hardware accelerators for API functions.
 *
 * These options specify the type of hardware accelerator to be used for 
 * parallel computation. The default is `MSZ_ACCELERATOR_CUDA`, which uses 
 * CUDA-based GPU acceleration.
 *
 * Values:
 * - `MSZ_ACCELERATOR_NONE`: Pure CPU execution.
 * - `MSZ_ACCELERATOR_OMP`: OpenMP-based CPU parallelism.
 * - `MSZ_ACCELERATOR_CUDA`: CUDA-based GPU acceleration.
 * - `MSZ_ACCELERATOR_HIP`: AMD GPU acceleration using HIP.
 * - `MSZ_ACCELERATOR_SYCL`: SYCL-based acceleration (e.g., for Intel GPUs).
 */
enum {
  MSZ_ACCELERATOR_NONE, // pure CPU
  MSZ_ACCELERATOR_OMP, // OpenMP
  MSZ_ACCELERATOR_CUDA, // CUDA GPU
  MSZ_ACCELERATOR_HIP,  // AMD GPU
  MSZ_ACCELERATOR_SYCL  // SYCL (Intel GPU)
};

// Struct for representing the edits
struct MSz_edit_t {
    uint32_t index; // index where edit is applied
    double offset;  // offset value corresponding to the index
};

// Enum for critical point types
enum {
    MSZ_CRITICAL_MINIMUM = 0,  // Local minimum
    MSZ_CRITICAL_MAXIMUM = 1,  // Local maximum
    MSZ_CRITICAL_SADDLE = 2    // Saddle point (for future use)
};

// Struct for representing critical points
struct MSz_critical_point_t {
    uint32_t index;      // Linear index of the critical point in the data array
    int x, y, z;         // 3D coordinates of the critical point (z=0 for 2D data)
    double value;        // Function value at the critical point
    uint8_t type;        // Type of critical point (MSZ_CRITICAL_MINIMUM, MSZ_CRITICAL_MAXIMUM, etc.)
};




  /**
  * @brief API for computing topology-preserving edits.
  * This API computes a series of edits to apply to decompressed data, ensuring the preservation 
  * of user-specified topological features. Supported features include local maxima, local minima, 
  * and separatrices connecting maxima and minima. The edits ensure accurate reconstruction 
  * of these features while maintaining the decompressed data within the prescribed error bound.
  * @param original_data: Pointer to the original data array.
  * @param decompressed_data: Pointer to the decompressed data array.
  * @param num_edits Output parameter to store the number of edits calculated.
  * @param edits Output parameter to store the resulting edits array. Caller is responsible for freeing this memory.
  * @param edited_decompressed_data: Pointer to an array where edited data will be stored (optional, can be nullptr).
  * @param preservation_options: Bitset of preservation options (e.g., MSZ_PRESERVE_MIN | MSZ_PRESERVE_MAX).
  * @param connectivity_type: Connectivity type specifier.
    (0: Piecewise linear connecti vity (e.g., 2D case: connects only up, down, left, right, up-right, and bottom-left
      1: Full connectivity (e.g., 2D: also all diagonal connections).
    ).
    - **DO NOT use this option if:**
  *          - `preserve_min` or `preserve_max` is 0.
  *          - `connection_type` is set to 1 (full connectivity).
  * @param W, H, D: Dimensions of the data.
  *          - `W`: Integer representing the width (x-dimension) of the data grid.
  *          - `H`: Integer representing the height (y-dimension) of the data grid.
  *          - `D`: Integer representing the depth (z-dimension) of the data grid.
  *                  - For 2D datasets, set `depth` to 1.
  * @param rel_err_bound: Relative error bound.
  * @param accelerator Hardware accelerator for computation:
  *        - `MSZ_ACCELERATOR_CUDA`: Use CUDA-based GPU acceleration.
  *        - `MSZ_ACCELERATOR_OMP`: Use OpenMP-based CPU parallelization.
  *        - `MSZ_ACCELERATOR_NONE`: Use pure CPU for computation.
  * @param device_id GPU device ID (used only if `accelerator` is `MSZ_ACCELERATOR_CUDA`).
  * @param num_omp_threads Number of threads (used only if `accelerator` is `MSZ_ACCELERATOR_OMP`).
  *
  * @return Returns `0` (MSZ_ERR_NO_ERROR) if the function executes successfully. 
  *         Other error codes indicate failures such as invalid inputs, resource limitations, 
  *         or unsupported accelerator types.
  *
  * @note Ensure that `original_data` and `decompressed_data` point to valid memory regions with dimensions matching `W x H x D`.

  * @example
  * For a 3D dataset with dimensions 100x100x100, you can calculate the edits as follows:
  * 
  * double* original_data = /* allocate and initialize the original dataset 
  * double* decompressed_data = /* allocate and initialize the decompressed dataset 
  * MSz_edit_t* edits = nullptr;
  * int num_edits = 0;
  * MSz_edit_t* edits = MSz_derive_edits(
  *                                   original_data,                // Input: original dataset
  *                                   decompressed_data,         // Input: decompressed dataset
  *                                   nullptr,                   // Optional: no edited data returned
  *                                   num_edits,                 // Output: number of edits
  *                                   &edits,                     // Output: edits array
  *                                   MSZ_PRESERVE_MIN | MSZ_PRESERVE_MAX, // Preserve minima and maxima
  *                                   0,                         // Piecewise connectivity
  *                                   100, 100, 100,             // W, H, D
  *                                   1e-3,                      // Relative error bound
  *                                   MSZ_ACCELERATOR_CUDA,              // Use CUDA for acceleration
  *                                   0,                                 // Use default GPU device
  *                                   0                                  // OpenMP threads not applicable
  * );
  */
  int MSz_derive_edits( // return 0 if success
      const double *original_data,   // Input: original data array
      const double *decompressed_data, // Input: decompressed data array
      double *edited_decompressed_data, // Output: edited data array (optional, can be nullptr)
      int &num_edits,                // Output: number of edits
      MSz_edit_t **edits,
      unsigned int preservation_options, //bitset for preservation options
      unsigned int connectivity_type,    // connectivity type specifier
      int W, int H, int D,           // Dimensions of the data
      double rel_err_bound,           // Relative error bound for edits
      int accelerator = MSZ_ACCELERATOR_CUDA,  // hardware accelerator
      int device_id = 0,               // GPU device ID (used if accelerator is CUDA)
      int num_omp_threads = 1        // Number of threads (used if accelerator is OMP)
  );

  /**
  * @brief API for counting topological distortions in the decompressed data.
  * This function identifies and counts discrepancies between the original data and decompressed data,
  * including false extrema (minima, maxima, and saddle points) and incorrectly labeled points in the Morse-Smale segmentation.
  * @param original_data Pointer to the original data array.
  * @param decompressed_data Pointer to the decompressed data array.
  * @param num_false_min Output parameter to store the number of false minima detected in the decompressed data.
  * @param num_false_max Output parameter to store the number of false maxima detected in the decompressed data.
  * @param num_false_saddle Output parameter to store the number of false saddle points detected in the decompressed data.
  * @param num_false_labels Output parameter to store the number of incorrectly labeled points in the Morse-Smale segmentation.
  * @param connectivity_type Connectivity type specifier:
  *        - `0`: Piecewise linear connectivity (e.g., in 2D: up, down, left, right, up-right, and bottom-left).
  *        - `1`: Full connectivity (e.g., in 2D: includes all connections including diagonals).
  * @param W, H, D Dimensions of the data grid:
  *        - `W`: Width of the data grid (x-dimension).
  *        - `H`: Height of the data grid (y-dimension).
  *        - `D`: Depth of the data grid (z-dimension). For 2D datasets, set `D` to 1.
  * @param accelerator Hardware accelerator for computation:
  *        - `MSZ_ACCELERATOR_CUDA`: Use CUDA-based GPU acceleration.
  *        - `MSZ_ACCELERATOR_OMP`: Use OpenMP-based CPU parallelization.
  *        - `MSZ_ACCELERATOR_NONE`: Use pure CPU for computation.
  * @param device_id GPU device ID (used only if `accelerator` is `MSZ_ACCELERATOR_CUDA`).
  * @param num_omp_threads Number of threads (used only if `accelerator` is `MSZ_ACCELERATOR_OMP`).
  *
  * @return Returns `0` (MSZ_ERR_NO_ERROR) if the function executes successfully. 
  *         Other error codes indicate failures such as invalid inputs or resource limitations.
  *
  * @note Ensure that `original_data` and `decompressed_data` point to valid memory regions of size `W x H x D`.
  *       This function does not modify the input arrays.
  *
  * @example
    * double* original_data = /* allocate and initialize the original dataset 
    * double* decompressed_data = /* allocate and initialize the decompressed dataset 
    * int status = MSz_count_faults(
    *     original_data,                   // Input: original dataset
    *     decompressed_data,               // Input: decompressed dataset
    *     num_false_min,                   // Output: false minima count
    *     num_false_max,                   // Output: false maxima count
    *     num_false_saddle,                // Output: false saddle points count
    *     num_false_labels,                // Output: mislabeled points count
    *     0,                               // Connectivity type: piecewise linear
    *     100, 100, 100,                   // Dimensions: W, H, D
    *     MSZ_ACCELERATOR_CUDA,            // Use CUDA for acceleration
    *     0,                               // Use default device
    *     0
    * );
    * 
    * if (status == 0) {
    *     std::cout << "Fault counting succeeded." << std::endl;
    *     std::cout << "False minima: " << num_false_min << std::endl;
    *     std::cout << "False maxima: " << num_false_max << std::endl;
    *     std::cout << "False saddle points: " << num_false_saddle << std::endl;
    *     std::cout << "False labels: " << num_false_labels << std::endl;
    * } else {
    *     std::cerr << "Error in fault counting. Error code: " << status << std::endl;
    * }
  */

  int MSz_count_faults(
      const double *original_data,      // Input: original data array
      const double *decompressed_data, // Input: decompressed data array
      int &num_false_min,              // Output: number of false minima
      int &num_false_max,              // Output: number of false maxima
      int &num_false_labels,           // Output: number of mislabeled points
      unsigned int connectivity_type,  // Connectivity type specifier
      int W, int H, int D,             // Dimensions of the data
      int accelerator = MSZ_ACCELERATOR_CUDA, // Hardware accelerator
      int device_id = 0,               // GPU device ID (used if accelerator is CUDA)
      int num_omp_threads = 1        // Number of threads (used if accelerator is OMP)
  );

  /**
  * @brief API for calculating the ratio of mislabeled points in the decompressed data.
  *
  * This function computes the ratio of mislabeled points (`num_false_labels`) relative to the 
  * total number of points in the dataset (`W x H x D`).
  *
  * @param num_false_labels The number of mislabeled points.
  * @param W The width (x-dimension) of the data grid.
  * @param H The height (y-dimension) of the data grid.
  * @param D The depth (z-dimension) of the data grid. For 2D datasets, set `D` to 1.
  *
  * @return Returns the ratio of mislabeled points, computed as:
  *         \f$ \text{ratio} = \frac{\text{num_false_labels}}{W \times H \times D} \f$.
  *         The return value is in the range `[0.0, 1.0]`.
  *
  * @note Ensure that `W`, `H`, and `D` are positive integers, and `num_false_labels` is non-negative.
  *       If the dataset dimensions are invalid (e.g., `W x H x D == 0`), the function may return `0.0`.
  *
  * @example
  * For a 3D dataset with dimensions 100x100x100 and 500 mislabeled points, the ratio can be computed as follows:
  *
  * 
  * int W = 100, H = 100, D = 100;
  * 
  * double ratio = MSz_false_label_ratio(num_false_labels, W, H, D);
  * 
  * std::cout << "False label ratio: " << ratio << std::endl;
  * 
  */

  double MSz_calculate_false_label_ratio(int num_false_labels,int W, int H, int D);
 

  /**
  * @brief API for losslessly compressing the topology-preserving edits using Zstandard (Zstd).
  *
  * This function compresses the indices and offsets from an array of topology-preserving edits 
  * (`MSz_edit_t`) into separate Zstandard-compressed buffers.
  *
  * @param num_edits The number of edits.
  * @param edits Pointer to an array of `MSz_edit_t`, where each edit contains an index and an offset.
  * @param compressed_buffer Output parameter to store the compressed buffer for edits. 
  *        The function allocates memory for the buffer, and the caller is responsible for freeing it.
  * @param compressed_size Output parameter to store the size of the compressed buffer (in bytes).
  *
  * @return Returns `MSZ_ERR_NO_ERROR` if the function executes successfully.
  *         Possible error codes include:
  *         - `MSZ_ERR_INVALID_INPUT`: Input parameters are invalid (e.g., `num_edits < 0` or `edits == nullptr`).
  *         - `MSZ_ERR_OUT_OF_MEMORY`: Memory allocation failed for compressed buffers.
  *         - `MSZ_ERR_EDITS_COMPRESSION_FAILED`: Compression failed due to Zstd errors.
  *
  * @note Ensure that `num_edits` is a nonnegative integer and `edits` points to valid memory. 
  *       The caller is responsible for freeing the memory allocated for `compressed_buffer` 
  *
  * @example
  * For a dataset with 3 topology-preserving edits, compress the edits as follows:

  * 
  * int num_edits = 3;
  * char *compressed__buffer = nullptr;
  * size_t compressed_size = 0;
  * 
  * int status = MSz_compress_edits_zstd(
  *     num_edits,                     // Number of edits
  *     edits,                         // Input edits array
  *     &compressed_buffer,            // Compressed index buffer
  *     compressed_size,               // Compressed size
  * );
  * 
  * if (status == MSZ_ERR_NO_ERROR) {
  *     std::cout << "Compression succeeded!" << std::endl;
  *     std::cout << "Compressed index size: " << compressed_size << " bytes" << std::endl;
  * 
  *     // Free allocated memory
  *     free(compressed_buffer);
  * } else {
  *     std::cerr << "Compression failed with error code: " << status << std::endl;
  * }
 */

  int MSz_compress_edits_zstd( // return MSZ_ERR_NO_ERROR if success
        int num_edits, // Number of edits
        MSz_edit_t *edits, // Input: Array of edits
        char **compressed_buffer, // Output: compressed buffer
        size_t &compressed_size // Output: size of compressed buffer
  );
  

  /**
 * @brief API for decompressing the topology-preserving edits.
 *
 * This function decompresses Zstandard-compressed buffers for indices and offsets into an array of 
 * topology-preserving edits (`MSz_edit_t`). 
 *
 * @param compressed_index_buffer Pointer to the buffer containing compressed indices.
 * @param compressed_offset_buffer Pointer to the buffer containing compressed offsets.
 * @param compressed_index_size The size of the compressed index buffer (in bytes).
 * @param compressed_offset_size The size of the compressed offset buffer (in bytes).
 * @param num_edits Output parameter to store the number of decompressed edits.
 * @param edits Output parameter to store a pointer to the array of decompressed edits (`MSz_edit_t`).
 *        The function allocates memory for the edits array, and the caller is responsible for freeing it.
 *
 * @return Returns `MSZ_ERR_NO_ERROR` if the function executes successfully.
 *         Possible error codes include:
 *         - `MSZ_ERR_INVALID_INPUT`: Input parameters are invalid (e.g., null pointers or size is `0`).
 *         - `MSZ_ERR_OUT_OF_MEMORY`: Memory allocation failed for decompressed edits.
 *         - `MSZ_ERR_EDITS_DECOMPRESSION_FAILED`: Decompression failed due to Zstd errors.
 *
 * @note 
 * - Ensure that `compressed_index_buffer` and `compressed_offset_buffer` point to valid Zstandard-compressed data.
 * - The caller is responsible for freeing the memory allocated for the `edits` array.
 *
 * @example
 * For a dataset with compressed index and offset buffers, decompress the edits as follows:

 * const char *compressed_index_buffer = /* Load or receive compressed index buffer;
 * const char *compressed_offset_buffer = /* Load or receive compressed offset buffer;
 * size_t compressed_index_size = /* Size of the compressed index buffer;
 * size_t compressed_offset_size = /* Size of the compressed offset buffer;
 * 
 * int num_edits = 0;
 * MSz_edit_t *edits = nullptr;
 * 
 * int status = MSz_decompress_edits(
 *     compressed_buffer,            // Input: compressed edits buffer
 *     compressed_size,              // Size of compressed buffer
 *     num_edits,                    // Output: number of decompressed edits
 *     &edits                        // Output: decompressed edits array
 * );
 * 
 * if (status == MSZ_ERR_NO_ERROR) {
 *     std::cout << "Decompression succeeded!" << std::endl;
 *     std::cout << "Number of edits: " << num_edits << std::endl;
 *     for (int i = 0; i < num_edits; ++i) {
 *         std::cout << "Edit " << i << ": Index = " << edits[i].index
 *                   << ", Offset = " << edits[i].offset << std::endl;
 *     }
 * 
 *     // Free allocated memory
 *     free(edits);
 * } else {
 *     std::cerr << "Decompression failed with error code: " << status << std::endl;
 * }

 */
  int MSz_decompress_edits_zstd( // return MSZ_ERR_NO_ERROR if success
      const char *compressed_buffer, // Input: pointer to the compressed buffer
      size_t compressed_size,        // Input: size of the compressed buffer (in bytes)
      int &num_edits,                // Output: number of decompressed edits
      MSz_edit_t **edits             // Output: pointer to an array of decompressed edits
  );


  /**
 * @brief API for applying topology-preserving edits to decompressed data.
 *
 * This function applies the computed edits to the decompressed data to ensure the preservation 
 * of topological features (e.g., minima, maxima, separatrices).
 *
 * @param decompressed_data Pointer to the decompressed data array.
 * @param num_edits Number of edits to apply.
 * @param edits Pointer to the array of edits (`MSz_edit_t`) to be applied.
 * @param W, H, D Dimensions of the data grid:
 *        - `W`: Width of the data grid (x-dimension).
 *        - `H`: Height of the data grid (y-dimension).
 *        - `D`: Depth of the data grid (z-dimension). For 2D datasets, set `D` to 1.
 * @param accelerator Hardware accelerator for computation:
 *        - `MSZ_ACCELERATOR_NONE`: Pure CPU execution.
 *        - `MSZ_ACCELERATOR_OMP`: OpenMP-based CPU parallelization.
 *        - `MSZ_ACCELERATOR_CUDA`: CUDA-based GPU acceleration.
 * @param device_id GPU device ID (used only if `accelerator` is `MSZ_ACCELERATOR_CUDA`).
 * @param num_omp_threads Number of threads (used only if `accelerator` is `MSZ_ACCELERATOR_OMP`).
 *
 * @return Returns `MSZ_ERR_NO_ERROR` if the function executes successfully.
 *         Possible error codes include:
 *         - `MSZ_ERR_INVALID_INPUT`: Input parameters are invalid.
 *         - `MSZ_ERR_OUT_OF_MEMORY`: Memory allocation failed.
 *         - `MSZ_ERR_UNKNOWN_ERROR`: An unknown error occurred during the application of edits.
 *
 * @note 
 * - Ensure that `decompressed_data` points to a valid memory region with dimensions `W x H x D`.
 * - Ensure that `edits` is a valid array with `num_edits` elements.
 */
  int MSz_apply_edits( // return MSZ_ERR_NO_ERROR if success
      double *decompressed_data,     // Input/Output: decompressed data to be modified
      int num_edits,                 // Input: number of edits to apply
      const MSz_edit_t *edits,       // Input: array of edits
      int W, int H, int D,           // Input: dimensions of the data
      int accelerator = MSZ_ACCELERATOR_NONE, // Input: hardware accelerator
      int device_id = 0,             // Input: GPU device ID (if using CUDA)
      int num_omp_threads = 1        // Input: number of threads (if using OpenMP)
  );


  /**
 * @brief API for extracting critical points from input data for visualization.
 *
 * This function identifies and extracts all critical points (local minima and maxima) from the input data,
 * returning them in separate arrays by type. Critical points are essential topological features that can be
 * used for visualization, analysis, and understanding the structure of scalar fields.
 *
 * @param data Pointer to the input data array.
 * @param num_minima Output parameter to store the number of minima found.
 * @param minima Output parameter to store a pointer to the array of minima critical points.
 *        The function allocates memory for the array, and the caller is responsible for freeing it.
 *        Will be set to nullptr if no minima are found.
 * @param num_maxima Output parameter to store the number of maxima found.
 * @param maxima Output parameter to store a pointer to the array of maxima critical points.
 *        The function allocates memory for the array, and the caller is responsible for freeing it.
 *        Will be set to nullptr if no maxima are found.
 * @param connectivity_type Connectivity type specifier:
 *        - `0`: Piecewise linear connectivity (e.g., in 2D: up, down, left, right, up-right, and bottom-left).
 *        - `1`: Full connectivity (e.g., in 2D: includes all diagonal connections).
 * @param W, H, D Dimensions of the data grid:
 *        - `W`: Width of the data grid (x-dimension).
 *        - `H`: Height of the data grid (y-dimension).
 *        - `D`: Depth of the data grid (z-dimension). For 2D datasets, set `D` to 1.
 * @param accelerator Hardware accelerator for computation:
 *        - `MSZ_ACCELERATOR_CUDA`: Use CUDA-based GPU acceleration.
 *        - `MSZ_ACCELERATOR_OMP`: Use OpenMP-based CPU parallelization.
 *        - `MSZ_ACCELERATOR_NONE`: Use pure CPU for computation.
 * @param device_id GPU device ID (used only if `accelerator` is `MSZ_ACCELERATOR_CUDA`).
 * @param num_omp_threads Number of threads (used only if `accelerator` is `MSZ_ACCELERATOR_OMP`).
 *
 * @return Returns `MSZ_ERR_NO_ERROR` if the function executes successfully.
 *         Possible error codes include:
 *         - `MSZ_ERR_INVALID_INPUT`: Input parameters are invalid.
 *         - `MSZ_ERR_OUT_OF_MEMORY`: Memory allocation failed.
 *         - `MSZ_ERR_UNKNOWN_ERROR`: An unknown error occurred.
 *
 * @note 
 * - Ensure that `data` points to a valid memory region with dimensions `W x H x D`.
 * - The caller is responsible for freeing the memory allocated for both `minima` and `maxima` using `free()`.
 * - Each critical point includes its linear index, 3D coordinates (x, y, z), function value, and type.
 * - Both minima and maxima are always extracted and returned in separate arrays.
 *
 * @example
 * For a 2D dataset with dimensions 100x100, extract all critical points:
 *
 * double* data = /* allocate and initialize the dataset;
 * int num_minima = 0, num_maxima = 0;
 * MSz_critical_point_t* minima = nullptr;
 * MSz_critical_point_t* maxima = nullptr;
 * 
 * int status = MSz_extract_critical_points(
 *     data,                             // Input: data array
 *     num_minima,                       // Output: number of minima
 *     &minima,                          // Output: minima array
 *     num_maxima,                       // Output: number of maxima
 *     &maxima,                          // Output: maxima array
 *     0,                                // Piecewise linear connectivity
 *     100, 100, 1,                      // W, H, D (2D data)
 *     MSZ_ACCELERATOR_CUDA,             // Use CUDA for acceleration
 *     0,                                // Use default GPU device
 *     1                                 // OpenMP threads not applicable
 * );
 * 
 * if (status == MSZ_ERR_NO_ERROR) {
 *     std::cout << "Found " << num_minima << " minima and " << num_maxima << " maxima." << std::endl;
 *     
 *     std::cout << "\\nMinima:" << std::endl;
 *     for (int i = 0; i < num_minima; ++i) {
 *         std::cout << "  [" << i << "] Position=(" << minima[i].x << ", "
 *                   << minima[i].y << ", " << minima[i].z << "), Value=" 
 *                   << minima[i].value << std::endl;
 *     }
 *     
 *     std::cout << "\\nMaxima:" << std::endl;
 *     for (int i = 0; i < num_maxima; ++i) {
 *         std::cout << "  [" << i << "] Position=(" << maxima[i].x << ", "
 *                   << maxima[i].y << ", " << maxima[i].z << "), Value=" 
 *                   << maxima[i].value << std::endl;
 *     }
 *     
 *     // Free allocated memory
 *     free(minima);
 *     free(maxima);
 * } else {
 *     std::cerr << "Failed to extract critical points. Error code: " << status << std::endl;
 * }
 */
  int MSz_extract_critical_points( // return MSZ_ERR_NO_ERROR if success
      const double *data,                    // Input: data array
      int &num_minima,                       // Output: number of minima
      MSz_critical_point_t **minima,         // Output: array of minima
      int &num_maxima,                       // Output: number of maxima
      MSz_critical_point_t **maxima,         // Output: array of maxima
      int &num_saddle_points,                // Output: number of saddle points
      MSz_critical_point_t **saddle_points,  // Output: array of saddle points
      unsigned int connectivity_type,        // Connectivity type specifier
      int W, int H, int D,                   // Dimensions of the data
      int accelerator = MSZ_ACCELERATOR_NONE, // Hardware accelerator
      int device_id = 0,                     // GPU device ID (used if accelerator is CUDA)
      int num_omp_threads = 1                // Number of threads (used if accelerator is OMP)
  );


#endif // _MSZ_API_H
