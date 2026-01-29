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

    // Struct
    py::class_<MSz_edit_t>(m, "Edit")
        .def(py::init<uint32_t, double>(), py::arg("index"), py::arg("offset"))
        .def_readwrite("index", &MSz_edit_t::index)
        .def_readwrite("offset", &MSz_edit_t::offset)
        .def("__repr__", [](const MSz_edit_t &e) {
            return "<msz.Edit index=" + std::to_string(e.index) + " offset=" + std::to_string(e.offset) + ">";
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
       py::arg("accelerator") = (int)MSZ_ACCELERATOR_CUDA,
       py::arg("device_id") = 0,
       py::arg("num_omp_threads") = 1);

    m.def("count_faults", [](py_array_double original, py_array_double decompressed,
                             unsigned int connectivity_type, int W, int H, int D,
                             int accelerator, int device_id, int num_omp_threads) {
        
        if (original.size() != (size_t)W*H*D || decompressed.size() != (size_t)W*H*D) {
            throw std::runtime_error("Array size does not match dimensions W*H*D");
        }

        int num_false_min = 0;
        int num_false_max = 0;
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
       py::arg("num_omp_threads") = 1);

    m.def("calculate_false_label_ratio", &MSz_calculate_false_label_ratio,
          py::arg("num_false_labels"), py::arg("W"), py::arg("H"), py::arg("D"));

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
    }, py::arg("edits"));

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
    }, py::arg("compressed"));

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
       py::arg("num_omp_threads") = 1);
}
