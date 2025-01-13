#ifndef MSZ_CUDA_H
#define MSZ_CUDA_H

#include <cstdint>
#include "api/MSz.h" 


extern "C" {


    /**
    * @brief Apply edits using CUDA acceleration.
    *
    * @param decompressed_data Pointer to the decompressed data array.
    * @param num_edits Number of edits to apply.
    * @param edits Pointer to the array of edits (`MSz_edit_t`) to be applied.
    * @param W, H, D Dimensions of the data grid.
    * @param device_id GPU device ID.
    * @return Returns `MSZ_ERR_NO_ERROR` on success or an error code on failure.
    */
    int MSz_apply_edits_cuda(
        double *decompressed_data,
        int num_edits,
        const MSz_edit_t *edits,
        int W, int H, int D,
        int device_id
    );

    int fix_process(std::vector<int> *a,std::vector<int> *b,
                std::vector<int> *c,std::vector<int> *d,
                std::vector<double> *input_data1,std::vector<double> *decp_data1,
                std::vector<int>* dec_label1,std::vector<int>* or_label1, 
                int width1, int height1, int depth1, 
                double bound1, 
                int preserve_min, int preserve_max, 
                int preserve_path, int neighbor_number, int device_id);
    
    int count_false_cases(std::vector<int> *a,std::vector<int> *b,
                std::vector<int> *c,std::vector<int> *d,
                std::vector<double> *input_data1,std::vector<double> *decp_data1,
                std::vector<int>* dec_label1,std::vector<int>* or_label1, 
                int width1, int height1, int depth1, int neighbor_number,
                int &wrong_min, int &wrong_max,  int &wrong_labels, int device_id);


}


#endif // MSZ_CUDA_H
