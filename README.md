# MSz: An Efficient Parallel Algorithm for Correcting Morse-Smale Segmentations in Error-Bounded Lossy Compressors

MSz is designed to preserve topological features such as local minima, maxima, and integral paths with user-defined connectivity modes for 2D/3D datasets.

## Requirements
- C++17 or later
- CMake 3.18 or later
- CUDA (if needed)
- OpenMP (if needed)

## Installation
### 1. Clone the Repository
```bash
git clone --recursive https://github.com/YuxiaoLi1234/MSz.git
cd MSz
```
### 2. Build the Program
```bash
mkdir build
cd build
cmake ..
make
```

### 3. Parameters
1. path/to/data,width,height,depth
   - Description: Path to the input dataset, followed by its dimensions (width, height, depth).
   - Example: path/to/your/data.bin,256,256,128
2. relative_error_bound
   - Description: Relative error bound for processing as a floating-point value.
   - Example: 0.01
3. compressor_type
   - Description: Compression library to use. Supported values:
       - sz3
       - zfp
   - Example: sz3
4. connection_type
   - Description: Connectivity type for the dataset:
       - 0: Piecewise linear connectivity (e.g., 2D case: connects only up, down, left, right, up-right, and bottom-left).
       - 1: Full connectivity (e.g., 2D: also all diagonal connections).
   - Example: 0
5. preserve_min
   - Description: Whether to preserve local minima (0 for no, 1 for yes).
   - Example: 1
6. preserve_max
   - Description: Whether to preserve local maxima (0 for no, 1 for yes).
   - Example: 0
7. preserve_integral_lines
   - Description: Whether to preserve integral lines (0 for no, 1 for yes).
       - DO NOT use this option if:
           - <preserve_min> or <preserve_max> are set to 0.
           - <neighbor_number> is set to 1.
   - Example: 0







