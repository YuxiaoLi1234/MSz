#ifndef MSZ_CUDA_H
#define MSZ_CUDA_H

#include <cstdint>
#include "api/MSz.h" 

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif // MSZ_CUDA_H
