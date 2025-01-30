# **# MSz: An Efficient Parallel Algorithm for Correcting Morse-Smale Segmentations in Error-Bounded Lossy Compressors**
<a href="./LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>

**MSZ** is designed to preserve topological features such as local minima, maxima, and integral paths with user-defined connectivity modes for 2D/3D datasets. The API supports **GPU acceleration (CUDA)** and **multi-threaded CPU execution (OpenMP)**.
---

## 📦 **Installation**
### **1️⃣ Build & Install**
```bash
git clone https://github.com/your-repo/MSz.git
cd MSz
mkdir build && cd build
cmake .. 
make -j$(nproc)
make install
```

## **MSZ CLI - Command Line Interface**

### **Usage**

```bash
./MSz_CLI <command> <arguments> --mode <none|omp|cuda>
```

### **Available Commands**
| **Command**          | **Description** |
|----------------------|----------------|
| `count_faults`      | Detects false minima, maxima, and incorrect segmentation. |
| `derive_edits`      | Computes topology-preserving edits for decompressed data. |
| `apply_edits`       | Applies topology-preserving edits to decompressed data. |

### **CLI Options**
| **Option**           | **Description** |
|----------------------|----------------|
| `<original_file>`      | Path to the original uncompressed data file. |
| `<decompressed_file>`  | Path to the decompressed data file to analyze or modify. |
| `<edits_file>`         | Path to the file containing computed edits (required for `apply_edits`). |
| `<width>`              | Width (x-dimension) of the dataset. |
| `<height>`             | Height (y-dimension) of the dataset. |
| `<depth>`              | Depth (z-dimension) of the dataset (set to 1 for 2D data). |
| `<connectivity>`       | Connectivity type:<br> `0` - Piecewise connectivity (e.g., 2D: up, down, left, right, upper-right, lower-left)<br> `1` - Full connectivity (e.g., 2D: includes diagonals). |
| `<error_bound>`        | Floating-point value specifying the error bound (e.g., `1E-3`, `0.01`) [Only for `derive_edits`]. |
| `<preserve_min>`       | Preserve minima (`1` = Yes, `0` = No) [Only for `derive_edits`]. |
| `<preserve_max>`       | Preserve maxima (`1` = Yes, `0` = No) [Only for `derive_edits`]. |
| `<preserve_path>`      | Preserve separatrices (`1` = Yes, `0` = No) [Only for `derive_edits` and `connectivity == 0`]. |
| `--mode <exec_mode>`   | Execution mode:<br> `none` - Pure CPU execution <br> `omp` - OpenMP parallel execution <br> `cuda` - GPU acceleration using CUDA |

---

### **Examples**

#### **Detect topology faults in a dataset**
```bash
./MSz_CLI count_faults original.bin decompressed.bin 100 100 100 0 --mode cuda
```
**Expected Output:**
```txt
Using CUDA execution.
Count faults succeeded.
False minima: x
False maxima: x
False labels: x
```

#### **Compute topology-preserving edits**
```bash
./MSz_CLI derive_edits original.bin decompressed.bin 100 100 100 0 1E-3 1 1 1 --mode omp
```
**Expected Output:**
```txt
Using OpenMP execution.
Derive edits succeeded. Number of edits: 35
Edits saved to edits.bin
```

#### **Apply topology-preserving edits**
```bash
./MSz_CLI apply_edits decompressed.bin edits.bin 100 100 100 --mode cuda
```
**Expected Output:**
```txt
Using CUDA execution.
Apply edits succeeded.
Edited data saved to edited_data.bin
```

---

## 🛠 **API Usage**

### **1️⃣ Compute Topology-Preserving Edits**
```cpp
int MSz_derive_edits(
    const double *original_data,   // Input: original data array
    const double *decompressed_data, // Input: decompressed data array
    double *edited_decompressed_data, // Output: edited data array (optional, can be nullptr)
    int &num_edits,                // Output: number of edits
    MSz_edit_t **edits,             // Output: array of computed edits
    unsigned int preservation_options, // Bitset for preservation options
    unsigned int connectivity_type,    // Connectivity type specifier (0 = piecewise, 1 = full connectivity)
    int W, int H, int D,           // Dimensions of the data (width, height, depth)
    double rel_err_bound,           // Relative error bound for edits
    int accelerator = MSZ_ACCELERATOR_CUDA,  // Hardware accelerator
    int device_id = 0,               // GPU device ID (used if accelerator is CUDA)
    int num_omp_threads = 1        // Number of threads (used if accelerator is OMP)
);
```

## 🛠 **API Usage**

### **1️⃣ Compute Topology-Preserving Edits**
```cpp
int MSz_derive_edits(
    const double *original_data,   // Input: original data array
    const double *decompressed_data, // Input: decompressed data array
    double *edited_decompressed_data, // Output: edited data array (optional, can be nullptr)
    int &num_edits,                // Output: number of edits
    MSz_edit_t **edits,             // Output: array of computed edits
    unsigned int preservation_options, // Bitset for preservation options
    unsigned int connectivity_type,    // Connectivity type specifier (0 = piecewise, 1 = full connectivity)
    int W, int H, int D,           // Dimensions of the data (width, height, depth)
    double rel_err_bound,           // Relative error bound for edits
    int accelerator = MSZ_ACCELERATOR_CUDA,  // Hardware accelerator
    int device_id = 0,               // GPU device ID (used if accelerator is CUDA)
    int num_omp_threads = 1        // Number of threads (used if accelerator is OMP) 
);
```
**Description**:  
Computes the necessary edits to preserve critical topological features while keeping decompressed data within the given error bound.

---

### **2️⃣ Count Topological Distortions**
```cpp
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
```
**Description**:  
Detects **false minima, maxima, and incorrect segmentation** in decompressed data.

---

### **3️⃣ Apply Topology-Preserving Edits**
```cpp
int MSz_apply_edits( // return MSZ_ERR_NO_ERROR if success
    double *decompressed_data,     // Input/Output: decompressed data to be modified
    int num_edits,                 // Input: number of edits to apply
    const MSz_edit_t *edits,       // Input: array of edits
    int W, int H, int D,           // Input: dimensions of the data
    int accelerator = MSZ_ACCELERATOR_NONE, // Input: hardware accelerator
    int device_id = 0,             // Input: GPU device ID (if using CUDA)
    int num_omp_threads = 1        // Input: number of threads (if using OpenMP)
);
```
**Description**:  
Applies computed **topology-preserving edits** to modify decompressed data.

---

### **4️⃣ Compress Edits using Zstd**
```cpp
int MSz_compress_edits_zstd( // return MSZ_ERR_NO_ERROR if success
    int num_edits, // Number of edits
    MSz_edit_t *edits, // Input: Array of edits
    char **compressed_buffer, // Output: compressed buffer
    size_t &compressed_size // Output: size of compressed buffer
);
```
**Description**:  
Losslessly compresses edit data using **Zstandard (Zstd)** to reduce storage and transmission overhead.

---

### **5️⃣ Decompress Edits**
```cpp
int MSz_decompress_edits_zstd( // return MSZ_ERR_NO_ERROR if success
    const char *compressed_buffer, // Input: pointer to the compressed buffer
    size_t compressed_size,        // Input: size of the compressed buffer (in bytes)
    int &num_edits,                // Output: number of decompressed edits
    MSz_edit_t **edits             // Output: pointer to an array of decompressed edits
);
```
**Description**:  
Restores topology-preserving edits from a **Zstd-compressed buffer**.


---

## ⚙️ **Performance Optimization**
- **Multi-threading support (OpenMP)**
- **GPU acceleration (CUDA)**

---

## 📩 **Contact**
For questions or contributions, please reach out via **[li.14025@osu.edu]**.
