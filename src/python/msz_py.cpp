#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "api/MSz.h"
#include <vector>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;

// Use C-style contiguous arrays to ensure memory layout matches C++ expectations
using py_array_double = py::array_t<double, py::array::c_style | py::array::forcecast>;

PYBIND11_MODULE(msz, m) {
    m.doc() = "MSz Python bindings";

    // Enums / Constants
    m.attr("PRESERVE_MIN") = (int)MSZ_PRESERVE_MIN;
    m.attr("PRESERVE_MAX") = (int)MSZ_PRESERVE_MAX;
    m.attr("PRESERVE_SADDLE") = (int)MSZ_PRESERVE_SADDLE;
    m.attr("PRESERVE_PATH") = (int)MSZ_PRESERVE_PATH;

    m.attr("ACCELERATOR_NONE") = (int)MSZ_ACCELERATOR_NONE;
    m.attr("ACCELERATOR_OMP") = (int)MSZ_ACCELERATOR_OMP;
    m.attr("ACCELERATOR_CUDA") = (int)MSZ_ACCELERATOR_CUDA;
    m.attr("ACCELERATOR_HIP") = (int)MSZ_ACCELERATOR_HIP;
    m.attr("ACCELERATOR_SYCL") = (int)MSZ_ACCELERATOR_SYCL;

    m.attr("ERR_NO_ERROR") = (int)MSZ_ERR_NO_ERROR;
    m.attr("ERR_INVALID_INPUT") = (int)MSZ_ERR_INVALID_INPUT;
    m.attr("ERR_INVALID_CONNECTIVITY_TYPE") = (int)MSZ_ERR_INVALID_CONNECTIVITY_TYPE;
    m.attr("ERR_NO_AVAILABLE_GPU") = (int)MSZ_ERR_NO_AVAILABLE_GPU;
    m.attr("ERR_OUT_OF_MEMORY") = (int)MSZ_ERR_OUT_OF_MEMORY;
    m.attr("ERR_UNKNOWN_ERROR") = (int)MSZ_ERR_UNKNOWN_ERROR;
    m.attr("ERR_EDITS_COMPRESSION_FAILED") = (int)MSZ_ERR_EDITS_COMPRESSION_FAILED;
    m.attr("ERR_EDITS_DECOMPRESSION_FAILED") = (int)MSZ_ERR_EDITS_DECOMPRESSION_FAILED;
    m.attr("ERR_NOT_IMPLEMENTED") = (int)MSZ_ERR_NOT_IMPLEMENTED;
    m.attr("ERR_INVALID_THREAD_COUNT") = (int)MSZ_ERR_INVALID_THREAD_COUNT;

    // Enums for critical point types
    m.attr("CRITICAL_MINIMUM") = (int)MSZ_CRITICAL_MINIMUM;
    m.attr("CRITICAL_MAXIMUM") = (int)MSZ_CRITICAL_MAXIMUM;
    m.attr("CRITICAL_SADDLE") = (int)MSZ_CRITICAL_SADDLE;

    // Structs
    py::class_<MSz_edit_t>(m, "Edit")
        .def(py::init<uint32_t, double>(), py::arg("index"), py::arg("offset"))
        .def_readwrite("index", &MSz_edit_t::index)
        .def_readwrite("offset", &MSz_edit_t::offset)
        .def("__repr__", [](const MSz_edit_t &e) {
            return "<msz.Edit index=" + std::to_string(e.index) + " offset=" + std::to_string(e.offset) + ">";
        });

    py::class_<MSz_critical_point_t>(m, "CriticalPoint")
        .def(py::init<>())
        .def_readwrite("index", &MSz_critical_point_t::index)
        .def_readwrite("x", &MSz_critical_point_t::x)
        .def_readwrite("y", &MSz_critical_point_t::y)
        .def_readwrite("z", &MSz_critical_point_t::z)
        .def_readwrite("value", &MSz_critical_point_t::value)
        .def_readwrite("type", &MSz_critical_point_t::type)
        .def("__repr__", [](const MSz_critical_point_t &cp) {
            std::string type_str = (cp.type == MSZ_CRITICAL_MINIMUM) ? "MIN" : 
                                   (cp.type == MSZ_CRITICAL_MAXIMUM) ? "MAX" : "SADDLE";
            return "<msz.CriticalPoint type=" + type_str + 
                   " pos=(" + std::to_string(cp.x) + "," + std::to_string(cp.y) + "," + std::to_string(cp.z) + ")" +
                   " value=" + std::to_string(cp.value) + ">";
        });

    // Functions
    m.def("derive_edits", [](py_array_double original, py_array_double decompressed, 
                             unsigned int preservation_options, unsigned int connectivity_type,
                             int W, int H, int D, double rel_err_bound,
                             int accelerator, int device_id, int num_omp_threads) {
        
        if (original.size() != (size_t)W*H*D || decompressed.size() != (size_t)W*H*D) {
            throw std::runtime_error("Array size does not match dimensions W*H*D");
        }

        int num_edits = 0;
        MSz_edit_t* edits_ptr = nullptr;
        
        const double* ptr_orig = original.data();
        const double* ptr_decp = decompressed.data();

        int status;
        {
            py::gil_scoped_release release;
            status = MSz_derive_edits(
                ptr_orig,
                ptr_decp,
                nullptr, // edited_decompressed_data
                num_edits,
                &edits_ptr,
                preservation_options,
                connectivity_type,
                W, H, D,
                rel_err_bound,
                accelerator,
                device_id,
                num_omp_threads
            );
        }

        std::vector<MSz_edit_t> edits_vec;
        if (status == MSZ_ERR_NO_ERROR && edits_ptr != nullptr) {
            edits_vec.assign(edits_ptr, edits_ptr + num_edits);
            free(edits_ptr);
        }

        return std::make_pair(status, edits_vec);
    }, py::arg("original"), py::arg("decompressed"), 
       py::arg("preservation_options"), py::arg("connectivity_type"),
       py::arg("W"), py::arg("H"), py::arg("D"), py::arg("rel_err_bound"),
       py::arg("accelerator") = (int)MSZ_ACCELERATOR_NONE,
       py::arg("device_id") = 0,
       py::arg("num_omp_threads") = 1,
       R"pbdoc(
        Derive topology-preserving edits between original and decompressed data.
        
        Parameters
        ----------
        original : numpy.ndarray
            Original data array
        decompressed : numpy.ndarray
            Decompressed data array (same shape as original)
        preservation_options : int
            Topology preservation mode: PRESERVE_MIN, PRESERVE_MAX, or PRESERVE_PATH
        connectivity_type : int
            0 for piecewise linear connectivity, 1 for full connectivity
        W, H, D : int
            Dimensions of the data grid (for 2D data, set D=1)
        rel_err_bound : float
            Relative error bound for topology preservation
        accelerator : int, optional
            Hardware accelerator: ACCELERATOR_NONE, ACCELERATOR_OMP, or ACCELERATOR_CUDA
        device_id : int, optional
            GPU device ID (used only with ACCELERATOR_CUDA)
        num_omp_threads : int, optional
            Number of OpenMP threads (used only with ACCELERATOR_OMP)
        
        Returns
        -------
        tuple
            Tuple of (status, edits) where:
            - status: int - Error code (ERR_NO_ERROR on success)
            - edits: list of Edit - List of edits to apply
       )pbdoc");

    m.def("count_faults", [](py_array_double original, py_array_double decompressed,
                             unsigned int connectivity_type, int W, int H, int D,
                             int accelerator, int device_id, int num_omp_threads) {
        
        if (original.size() != (size_t)W*H*D || decompressed.size() != (size_t)W*H*D) {
            throw std::runtime_error("Array size does not match dimensions W*H*D");
        }

        int num_false_min = 0;
        int num_false_max = 0;
        int num_false_saddles = 0;
        int num_false_labels = 0;

        const double* ptr_orig = original.data();
        const double* ptr_decp = decompressed.data();

        int status;
        {
            py::gil_scoped_release release;
            status = MSz_count_faults(
                ptr_orig,
                ptr_decp,
                num_false_min,
                num_false_max,
                num_false_labels,
                connectivity_type,
                W, H, D,
                accelerator,
                device_id,
                num_omp_threads
            );
        }

        return py::dict("status"_a=status, "num_false_min"_a=num_false_min, 
                        "num_false_max"_a=num_false_max, "num_false_labels"_a=num_false_labels);
    }, py::arg("original"), py::arg("decompressed"), py::arg("connectivity_type"),
       py::arg("W"), py::arg("H"), py::arg("D"),
       py::arg("accelerator") = (int)MSZ_ACCELERATOR_CUDA,
       py::arg("device_id") = 0,
       py::arg("num_omp_threads") = 1,
       R"pbdoc(
        Count topology faults (false minima, maxima, and labels) between original and decompressed data.
        
        Parameters
        ----------
        original : numpy.ndarray
            Original data array
        decompressed : numpy.ndarray
            Decompressed data array (same shape as original)
        connectivity_type : int
            0 for piecewise linear connectivity, 1 for full connectivity
        W, H, D : int
            Dimensions of the data grid (for 2D data, set D=1)
        accelerator : int, optional
            Hardware accelerator: ACCELERATOR_NONE, ACCELERATOR_OMP, or ACCELERATOR_CUDA
        device_id : int, optional
            GPU device ID (used only with ACCELERATOR_CUDA)
        num_omp_threads : int, optional
            Number of OpenMP threads (used only with ACCELERATOR_OMP)
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'status': int - Error code (ERR_NO_ERROR on success)
            - 'num_false_min': int - Number of false minima
            - 'num_false_max': int - Number of false maxima
            - 'num_false_labels': int - Total number of mislabeled points
       )pbdoc");

    m.def("calculate_false_label_ratio", &MSz_calculate_false_label_ratio,
          py::arg("num_false_labels"), py::arg("W"), py::arg("H"), py::arg("D"),
          R"pbdoc(
        Calculate the ratio of false labels to total data points.
        
        Parameters
        ----------
        num_false_labels : int
            Number of mislabeled points
        W, H, D : int
            Dimensions of the data grid
        
        Returns
        -------
        float
            Ratio of false labels (num_false_labels / (W * H * D))
       )pbdoc");

    m.def("compress_edits_zstd", [](std::vector<MSz_edit_t> edits) {
        char* compressed_buffer = nullptr;
        size_t compressed_size = 0;
        int status = MSz_compress_edits_zstd(
            (int)edits.size(),
            edits.data(),
            &compressed_buffer,
            compressed_size
        );

        py::bytes result;
        if (status == MSZ_ERR_NO_ERROR && compressed_buffer != nullptr) {
            result = py::bytes(compressed_buffer, compressed_size);
            free(compressed_buffer);
        }
        return std::make_pair(status, result);
    }, py::arg("edits"),
       R"pbdoc(
        Compress a list of edits using Zstandard compression.
        
        Parameters
        ----------
        edits : list of Edit
            List of edits to compress
        
        Returns
        -------
        tuple
            Tuple of (status, compressed_data) where:
            - status: int - Error code (ERR_NO_ERROR on success)
            - compressed_data: bytes - Compressed binary data
        
        Examples
        --------
        >>> status, edits = msz.derive_edits(original, decompressed, ...)
        >>> status, compressed = msz.compress_edits_zstd(edits)
       )pbdoc");

    m.def("decompress_edits_zstd", [](py::bytes compressed) {
        std::string s = compressed;
        int num_edits = 0;
        MSz_edit_t* edits_ptr = nullptr;
        int status = MSz_decompress_edits_zstd(
            s.data(),
            s.size(),
            num_edits,
            &edits_ptr
        );

        std::vector<MSz_edit_t> edits_vec;
        if (status == MSZ_ERR_NO_ERROR && edits_ptr != nullptr) {
            edits_vec.assign(edits_ptr, edits_ptr + num_edits);
            free(edits_ptr);
        }
        return std::make_pair(status, edits_vec);
    }, py::arg("compressed"),
       R"pbdoc(
        Decompress edits from Zstandard compressed binary data.
        
        Parameters
        ----------
        compressed : bytes
            Compressed binary data from compress_edits_zstd()
        
        Returns
        -------
        tuple
            Tuple of (status, edits) where:
            - status: int - Error code (ERR_NO_ERROR on success)
            - edits: list of Edit - Decompressed list of edits
        
        Examples
        --------
        >>> status, edits = msz.decompress_edits_zstd(compressed_data)
       )pbdoc");

    m.def("apply_edits", [](py_array_double decompressed, std::vector<MSz_edit_t> edits,
                            int W, int H, int D, int accelerator, int device_id, int num_omp_threads) {
        
        if (decompressed.size() != (size_t)W*H*D) {
            throw std::runtime_error("Array size does not match dimensions W*H*D");
        }

        double* ptr_decp = const_cast<double*>(decompressed.data());
        int status;
        {
            py::gil_scoped_release release;
            status = MSz_apply_edits(
                ptr_decp,
                (int)edits.size(),
                edits.data(),
                W, H, D,
                accelerator,
                device_id,
                num_omp_threads
            );
        }
        return status;
    }, py::arg("decompressed"), py::arg("edits"), py::arg("W"), py::arg("H"), py::arg("D"),
       py::arg("accelerator") = (int)MSZ_ACCELERATOR_NONE,
       py::arg("device_id") = 0,
       py::arg("num_omp_threads") = 1,
       R"pbdoc(
        Apply topology-preserving edits to decompressed data (in-place modification).
        
        Parameters
        ----------
        decompressed : numpy.ndarray
            Decompressed data array to modify (will be modified in-place)
        edits : list of Edit
            List of edits to apply
        W, H, D : int
            Dimensions of the data grid (for 2D data, set D=1)
        accelerator : int, optional
            Hardware accelerator: ACCELERATOR_NONE, ACCELERATOR_OMP, or ACCELERATOR_CUDA
        device_id : int, optional
            GPU device ID (used only with ACCELERATOR_CUDA)
        num_omp_threads : int, optional
            Number of OpenMP threads (used only with ACCELERATOR_OMP)
        
        Returns
        -------
        int
            Error code (ERR_NO_ERROR on success)
        
        Examples
        --------
        >>> status, edits = msz.derive_edits(original, decompressed, ...)
        >>> status = msz.apply_edits(decompressed, edits, 100, 100, 1)
       )pbdoc");

    m.def("extract_critical_points", [](py_array_double data, unsigned int connectivity_type,
                                        int W, int H, int D, int accelerator, 
                                        int device_id, int num_omp_threads) {
        
        if (data.size() != (size_t)W*H*D) {
            throw std::runtime_error("Array size does not match dimensions W*H*D");
        }

        int num_minima = 0;
        int num_maxima = 0;
        int num_saddle = 0;
        MSz_critical_point_t* minima_ptr = nullptr;
        MSz_critical_point_t* maxima_ptr = nullptr;
        MSz_critical_point_t* saddle_ptr = nullptr;
        
        const double* ptr_data = data.data();

        int status;
        {
            py::gil_scoped_release release;
            status = MSz_extract_critical_points(
                ptr_data,
                num_minima,
                &minima_ptr,
                num_maxima,
                &maxima_ptr,
                num_saddle,
                &saddle_ptr,
                connectivity_type,
                W, H, D,
                accelerator,
                device_id,
                num_omp_threads
            );
        }

        std::vector<MSz_critical_point_t> minima_vec;
        std::vector<MSz_critical_point_t> maxima_vec;
        std::vector<MSz_critical_point_t> saddle_vec;
        
        if (status == MSZ_ERR_NO_ERROR) {
            if (minima_ptr != nullptr) {
                minima_vec.assign(minima_ptr, minima_ptr + num_minima);
                free(minima_ptr);
            }
            if (maxima_ptr != nullptr) {
                maxima_vec.assign(maxima_ptr, maxima_ptr + num_maxima);
                free(maxima_ptr);
            }
            if (saddle_ptr != nullptr) {
                saddle_vec.assign(saddle_ptr, saddle_ptr + num_saddle);
                free(saddle_ptr);
            }
        }

        return py::dict("status"_a=status, "minima"_a=minima_vec, "maxima"_a=maxima_vec, "saddles"_a=saddle_vec);
    }, py::arg("data"), py::arg("connectivity_type"),
       py::arg("W"), py::arg("H"), py::arg("D"),
       py::arg("accelerator") = (int)MSZ_ACCELERATOR_NONE,
       py::arg("device_id") = 0,
       py::arg("num_omp_threads") = 1,
       R"pbdoc(
        Extract critical points (minima, maxima, and saddle points) from input data.
        
        Parameters
        ----------
        data : numpy.ndarray
            Input data array of shape (W, H, D) or (W*H*D,)
        connectivity_type : int
            0 for piecewise linear connectivity, 1 for full connectivity
        W, H, D : int
            Dimensions of the data grid (for 2D data, set D=1)
        accelerator : int, optional
            Hardware accelerator: ACCELERATOR_NONE, ACCELERATOR_OMP, or ACCELERATOR_CUDA
        device_id : int, optional
            GPU device ID (used only with ACCELERATOR_CUDA)
        num_omp_threads : int, optional
            Number of OpenMP threads (used only with ACCELERATOR_OMP)
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'status': int - Error code (ERR_NO_ERROR on success)
            - 'minima': list of CriticalPoint - List of minimum critical points
            - 'maxima': list of CriticalPoint - List of maximum critical points
            - 'saddles': list of CriticalPoint - List of saddle points
        
        Examples
        --------
        >>> import msz
        >>> import numpy as np
        >>> data = np.fromfile("../examples/datasets/grid100x100.bin")
        >>> result = msz.extract_critical_points(data, 0, 100, 100, 1)
        >>> print(f"Found {len(result['minima'])} minima, {len(result['maxima'])} maxima, and {len(result['saddles'])} saddles")
       )pbdoc");
}
