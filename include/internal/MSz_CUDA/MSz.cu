#include <iostream>
#include <float.h> 
#include <cublas_v2.h>
#include <fstream>
#include <sstream>
#include "../../api/MSz.h"
#include <vector>
#include <cstdlib>
#include <stdio.h>
#include <unordered_map>
#include <random>
#include <iostream>
#include <cstring> 
#include <chrono> 
#include <cuda_runtime.h>
#include <string>
#include <unordered_set>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <utility>
#include <iomanip>
#include <chrono>
#include <thrust/device_vector.h>
using std::count;
using std::cout;
using std::endl;


__device__ double* decp_data;
__device__ double* decp_data_copy ;
__device__ int directions1[78] = {
    1, 0, 0, -1, 0, 0,   
    0, 1, 0, 0, -1, 0,
    0, 0, 1, 0, 0, -1,
    1, 1, 0, 1, -1, 0,  
    -1, 1, 0, -1, -1, 0,
    1, 0, 1, 1, 0, -1,
    -1, 0, 1, -1, 0, -1,
    0, 1, 1, 0, 1, -1,
    0, -1, 1, 0, -1, -1,
    1, 1, 1, 1, 1, -1,  
    1, -1, 1, 1, -1, -1,
    -1, 1, 1, -1, 1, -1,
    -1, -1, 1, -1, -1, -1
};

__device__ int width;
__device__ int height;
__device__ int depth;
__device__ int num;
__device__ int* adjacency;
__device__ double* d_deltaBuffer1;
__device__ int* number_array;
__device__ int* all_max; 
__device__ int* all_min;
__device__ int* all_p_max; 
__device__ int* all_p_min;
__device__ int* unsigned_n;
__device__ int count_max;
__device__ int count_min;
__device__ int count_f_max;
__device__ int count_f_min;
__device__ int count_p_max;
__device__ int count_p_min;
__device__ int count_false_label;
__device__ int* maxi;

__device__ double bound;
__device__ int edit_count;
__device__ int* or_maxi;
__device__ int* or_mini;
__device__ double* d_deltaBuffer;
__device__ int* id_array;
__device__ int* or_label;
__device__ int* dec_label;

__device__ double* input_data;
__device__ int* de_direction_as;
__device__ int* de_direction_ds;
__device__ int maxNeighbors = 12;

__device__ int direction_to_index_mapping_cuda[26][3] = 
{
    {1, 0, 0}, {-1, 0, 0},   
    {0, 1, 0}, {0, -1, 0},
    {0, 0, 1}, {0, 0, -1},
    {1, 1, 0}, {1, -1, 0},  
    {-1, 1, 0}, {-1, -1, 0},
    {1, 0, 1}, {1, 0, -1},
    {-1, 0, 1}, {-1, 0, -1},
    {0, 1, 1}, {0, 1, -1},
    {0, -1, 1}, {0, -1, -1},
    {1, 1, 1}, {1, 1, -1},  
    {1, -1, 1}, {1, -1, -1},
    {-1, 1, 1}, {-1, 1, -1},
    {-1, -1, 1}, {-1, -1, -1}
};   


__device__ int getDirection(int x, int y, int z){
    
    for (int i = 0; i < maxNeighbors; ++i) {
        if (direction_to_index_mapping_cuda[i][0] == x && direction_to_index_mapping_cuda[i][1] == y && direction_to_index_mapping_cuda[i][2] == z) {
            return i+1;  
        }
    }
    return -1;  


}


__device__ int from_direction_to_index(int cur, int direc){
    
    if (direc==-1) return cur;
    int x = cur % width;
    int y = (cur / width) % height;
    int z = (cur/(width * height))%depth;
    
    if (direc >= 1 && direc <= maxNeighbors) {
        int delta_row = direction_to_index_mapping_cuda[direc-1][0];
        int delta_col = direction_to_index_mapping_cuda[direc-1][1];
        int delta_dep = direction_to_index_mapping_cuda[direc-1][2];
        
        
        int next_row = x + delta_row;
        int next_col = y + delta_col;
        int next_dep = z + delta_dep;
        
        return next_row + next_col * width + next_dep* (height * width);
    }
    else {
        return -1;
    }
    // return 0;
};

__global__ void find_direction (int type=0){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index>=num){
        return;
    }
    
    double *data;
    int *direction_as;
    int *direction_ds;
    if(type==0){
        data = decp_data;
        direction_as = de_direction_as;
        direction_ds = de_direction_ds;
    }
    else{
        data = input_data;
        direction_as = or_maxi;
        direction_ds = or_mini;
    }
    
        
        
    int largetst_index = index;

    
    for(int j =0;j<maxNeighbors;++j){
        int i = adjacency[index*maxNeighbors+j];
        
        if(i==-1){
            continue;
        }
        
        if((data[i]>data[largetst_index] or (data[i]==data[largetst_index] and i>largetst_index))){
            
            largetst_index = i;
            // }
            
        };
    };
    int x_diff = (largetst_index % width) - (index % width);
    int y_diff = (largetst_index / (width)) % height - (index / (width)) % height;
    int z_diff = (largetst_index /(width * height)) % depth - (index /(width * height)) % depth;
    
    direction_as[index] = getDirection(x_diff, y_diff,z_diff);
    largetst_index = index;
    for(int j =0;j<maxNeighbors;++j){
        int i = adjacency[index*maxNeighbors+j];
        
        if(i==-1){
            continue;
        }
        
        if((data[i]<data[largetst_index] or (data[i]==data[largetst_index] and i<largetst_index))){
            
            largetst_index = i;
        };
    };
    
    
    x_diff = (largetst_index % width) - (index % width);
    y_diff = (largetst_index / (width)) % height - (index / (width)) % height;
    z_diff = (largetst_index /(width * height)) % depth - (index /(width * height)) % depth;
    direction_ds[index] = getDirection(x_diff, y_diff,z_diff);
    
    return;

};

__global__ void iscriticle(){
        
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        
        if(i>=num){
            
            return;
        }
        
        bool is_maxima = true;
        bool is_minima = true;
        
        for (int index=0;index<maxNeighbors;index++) {
            int j = adjacency[i*maxNeighbors+index];
            if(j==-1){
                continue;
            }
              
            if (decp_data[j] > decp_data[i]) {
                
                is_maxima = false;
                
                break;
            }
            else if(decp_data[j] == decp_data[i] and j>i){
                is_maxima = false;
                break;
            }
        }
        for (int index=0;index< maxNeighbors;index++) {
            int j = adjacency[i*maxNeighbors+index];
            if(j==-1){
                continue;
            }
            if (decp_data[j] < decp_data[i]) {
                is_minima = false;
                break;
            }
            else if(decp_data[j] == decp_data[i] and j<i){
                is_minima = false;
                break;
            }
        }
        
        
        if((is_maxima && or_maxi[i]!=-1) or (!is_maxima && or_maxi[i]==-1)){
            int idx_fp_max = atomicAdd(&count_f_max, 1);
            
            all_max[idx_fp_max] = i;
            
        }
        
        else if ((is_minima && or_mini[i]!=-1) or (!is_minima && or_mini[i]==-1)) {
            int idx_fp_min = atomicAdd(&count_f_min, 1);
            
            all_min[idx_fp_min] = i;
            
        } 
        
       
        
}

__global__ void get_wrong_index_path(){

    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
        
    if(i>=num){
        
        return;
    }
    
    if (or_label[i * 2 + 1] != dec_label[i * 2 + 1]) {
        int idx_fp_max = atomicAdd(&count_p_max, 1);
        all_p_max[idx_fp_max] = i;
            
    }
    if (or_label[i * 2] != dec_label[i * 2]) {
        int idx_fp_min = atomicAdd(&count_p_min, 1);
        all_p_min[idx_fp_min] = i;
    }
    
    

    return;
};

__global__ void get_wrong_index_count(){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i>=num){
        
        return;
    }
    
    if (or_label[i * 2 + 1] != dec_label[i * 2 + 1]) {
        atomicAdd(&count_false_label, 1);
    }
    else if (or_label[i * 2] != dec_label[i * 2]) {
        atomicAdd(&count_false_label, 1);
    }
    
    

    return;
};


__global__ void computeAdjacency() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num) {
        
        int y = (i / (width)) % height; // Get the x coordinate
        int x = i % width; // Get the y coordinate
        int z = (i / (width * height)) % depth;
        int neighborIdx = 0;
        
        for (int d = 0; d < maxNeighbors; d++) {
            
            int dirX = directions1[d * 3];     
            int dirY = directions1[d * 3 + 1]; 
            int dirZ = directions1[d * 3 + 2]; 
            int newX = x + dirX;
            int newY = y + dirY;
            int newZ = z + dirZ;
            int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
            
            if (newX >= 0 && newX < width && newY >= 0 && newY < height && r < width*height*depth && newZ<depth && newZ>=0 ) {
                
                adjacency[i * maxNeighbors + neighborIdx] = r;
                neighborIdx++;

            }
        }

        // Fill the remaining slots with -1 or another placeholder value
        
        for (int j = neighborIdx; j < maxNeighbors; ++j) {
            adjacency[i * maxNeighbors + j] = -1;
        }
    }
}



__device__ double atomicCASDouble(double* address, double val) {
   
    uint64_t* address_as_ull = (uint64_t*)address;
    uint64_t old_val_as_ull = *address_as_ull;
    uint64_t new_val_as_ull = __double_as_longlong(val);
    uint64_t assumed;


    assumed = old_val_as_ull;
    
    old_val_as_ull = atomicCAS((unsigned long long int*)address_as_ull, (unsigned long long int)assumed, (unsigned long long int)new_val_as_ull);
    return __longlong_as_double(old_val_as_ull);
}

__device__ int swap(int index, double delta){
    int update_successful = 0;
    
    while (update_successful==0) {
        double current_value = d_deltaBuffer[index];
        if (-delta > current_value) {
            double swapped = atomicCASDouble(&d_deltaBuffer[index], delta);
            if (swapped == current_value) {
                update_successful = 1;
                
            } 
        } else {
            update_successful = 1; 
        }
    }
}

__global__ void fix_maxi_critical(int direction, int cnt){
    int index_f = blockIdx.x * blockDim.x + threadIdx.x;
        
    int index;
    int next_vertex;

    if (direction == 0 && index_f<count_f_max){
        
        index = all_max[index_f];
        // if vertex is a regular point.
        if (or_maxi[index]!=-1){
            
            // find its largest neighbor
            
            next_vertex = from_direction_to_index(index,or_maxi[index]);
            
            double d = ((input_data[index] - bound) + decp_data[index]) / 2.0 - decp_data[index];
            
            if(decp_data[index]<decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index<next_vertex)){
                return;
            }

            
            double oldValue = d_deltaBuffer[index];
            
            if (d > oldValue) {
                swap(index, d);
            }  

            return;
            
            
            
        
        }
        else{
            // if is a maximum in the original data;
            
            int largest_index = from_direction_to_index(index, de_direction_as[index]);
            
            if(decp_data[index]>decp_data[largest_index] or(decp_data[index]==decp_data[largest_index] and index>largest_index)){
                return;
            }

            double d = ((input_data[largest_index] - bound) + decp_data[largest_index]) / 2.0 - decp_data[largest_index];
            
            double oldValue = d_deltaBuffer[largest_index];
            if (d > oldValue) {
                swap(largest_index, d);
            }  

            return;
        }
        
        
    
    }
    
    else if (direction != 0 && index_f<count_f_min){
        index = all_min[index_f];
        
        if (or_mini[index]!=-1){
           
            
            int next_vertex= from_direction_to_index(index,or_mini[index]);
            

            double d = ((input_data[next_vertex] - bound) + decp_data[index]) / 2.0 - decp_data[next_vertex];
            
            if(decp_data[index]>decp_data[next_vertex] or (decp_data[index]==decp_data[next_vertex] and index>next_vertex)){
                return;
            }

            double oldValue = d_deltaBuffer[next_vertex];
            if (d > oldValue) {
                swap(next_vertex, d);
            }  

            return;
       
        
        }
    
        else{
            
            int largest_index = from_direction_to_index(index,de_direction_ds[index]);
            
            if(decp_data[index]<decp_data[largest_index] or (decp_data[index]==decp_data[largest_index] and index<largest_index)){
                
                return;
            }
            
            double d = ((input_data[index] - bound) + decp_data[index]) / 2.0 - decp_data[index];
            
            double oldValue = d_deltaBuffer[index];
            if (d > oldValue) {
                swap(index, d);
            }  

            return; 
        }

        
    }    
    

    

    return;
}


__global__ void initializeKernel(double value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<num){
        d_deltaBuffer[tid] = -4.0 * bound;
    }

}

__global__ void fixpath(int direction){
    int index_f = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(direction == 0){
        if(index_f<count_p_max){

        
        int index = all_p_max[index_f];
        int cur = index;
        while (or_maxi[cur] == de_direction_as[cur]){
            int next_vertex =  from_direction_to_index(cur,de_direction_as[cur]);
            
            if(de_direction_as[cur]==-1 && next_vertex == cur){
                cur = -1;
                break;
            }
            if(next_vertex == cur){
                cur = next_vertex;
                break;
            };
            
            cur = next_vertex;
        }

        int start_vertex = cur;
        
        
        if (start_vertex==-1) return;
        else{
            
            int false_index= from_direction_to_index(cur,de_direction_as[cur]);
            int true_index= from_direction_to_index(cur, or_maxi[cur]);
            if(false_index==true_index) return;

            double d = ((input_data[false_index] - bound) + decp_data[false_index]) / 2.0 - decp_data[false_index];
             
            double oldValue = d_deltaBuffer[false_index];
            if (d > oldValue) {
                swap(false_index, d);
            }  

            return;
        }
        }
    }

    else 
    {
        if(index_f<count_p_min){
            
        int index = all_p_min[index_f];
        int cur = index;
        
        
        while (or_mini[cur] == de_direction_ds[cur]){
            
            int next_vertex = from_direction_to_index(cur,de_direction_ds[cur]);
            
            
            if (next_vertex == cur){
                cur = next_vertex;
                break;
            }
            cur = next_vertex;

            
                
        }
    
        int start_vertex = cur;
        
        if (start_vertex==-1) return;
        
        else{
            
            int false_index= from_direction_to_index(cur,de_direction_ds[cur]);
            int true_index= from_direction_to_index(cur, or_mini[cur]);
            if(false_index==true_index) return;

            double d = ((input_data[true_index] - bound) + decp_data[true_index]) / 2.0 - decp_data[true_index];
            double oldValue = d_deltaBuffer[true_index];
            if (d > oldValue) {
                swap(true_index, d);
            }  

            return;
        }
    }
    }
    return;
};

__global__ void applyDeltaBuffer() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num) {
        if(d_deltaBuffer[tid] > -4.0 * bound){
            
            if(abs(d_deltaBuffer[tid]) > 1e-15) decp_data[tid] += d_deltaBuffer[tid];
            else decp_data[tid] = input_data[tid] - bound;
        }

        
    }
    
}


__global__ void getlabel(int *un_sign_ds, int *un_sign_as, int type=0){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int *direction_as;
    int *direction_ds;
    int *label;
    
    if(i>=num){
        return;
    }
    
    if(type==0){
        direction_as = de_direction_as;
        direction_ds = de_direction_ds;
        label = dec_label;
    }
    else{
        direction_as = or_maxi;
        direction_ds = or_mini;
        label = or_label;
    }
    
    int cur = label[i*2+1];
    
    
        int next_vertex;
        
        if (cur!=-1 and direction_as[cur]!=-1){
            
            int direc = direction_as[cur];
            
            
            next_vertex = from_direction_to_index(cur, direc);
            
            
            if(label[next_vertex*2+1] == -1){
                label[i*2+1] = next_vertex;
                
            }
            
            else{
                
                label[i*2+1] = label[next_vertex*2+1];
                
                
            }
            
            if (direction_as[label[i*2+1]] != -1){
                
                *un_sign_as+=1;  
                
            }
            
        }
    
    
    
    
        cur = label[i*2];
        int next_vertex1;
        
        
        if (cur!=-1 and label[cur*2]!=-1){
            
            int direc = direction_ds[cur];
            
            next_vertex1 = from_direction_to_index(cur, direc);
            
            if(label[next_vertex1*2] == -1){
                label[i*2] = next_vertex1;
                
            }
            
            else if(label[label[next_vertex1*2]*2] == -1){
                label[i*2] = label[next_vertex1*2];  
            }
            
            else if(direction_ds[i]!=-1){
               
                if(label[next_vertex1*2]!=-1){
                    label[i*2] = label[next_vertex1*2];
                }
                
                else{

                    label[i*2] = next_vertex1;
                }
                
                
            }
            
            if (direction_ds[label[i*2]]!=-1){
                *un_sign_ds+=1;
                }
            } 
        
        
    return;

}


__global__ void initializeWithIndex(int size, int type=0) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int* label;
    if (index < size) {
        int *direction_ds;
        int *direction_as;
        if(type==0){
            direction_ds = de_direction_ds;
            direction_as = de_direction_as;
            label = dec_label;
        }
        else{
            direction_ds = or_mini;
            direction_as = or_maxi;
            label = or_label;
        }

        if(direction_ds[index]!=-1){
            label[index*2] = index;
            
        }
        else{
            label[index*2] = -1;
        }

        if(direction_as[index]!=-1){
            label[index*2+1] = index;
        }
        else{
            label[index*2+1] = -1;
        }
    }
}

__global__ void change_mode(int neighbor_number)
{
    if(neighbor_number != 0)
    {
        maxNeighbors = 26;
    }
}


int fix_process(std::vector<int> *a,std::vector<int> *b,
                std::vector<int> *c,std::vector<int> *d,
                std::vector<double> *input_data1,std::vector<double> *decp_data1,
                std::vector<int>* dec_label1,std::vector<int>* or_label1, 
                int width1, int height1, int depth1, 
                double bound1, 
                int preserve_min, int preserve_max, 
                int preserve_path, int neighbor_number, int device_id){
    cudaError_t cudaStatus = cudaSetDevice(device_id);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_NO_AVAILABLE_GPU;
    }
    int *temp, *temp1, *d_data, *or_l, *dec_l;
    
    double *temp3, *temp4;
    int max_n = 12;
    if(neighbor_number!=0) max_n = 26;
    int num1 = width1*height1*depth1;
    
    int h_un_sign_as = num1;
    int h_un_sign_ds = num1;
  
    
    int *un_sign_as;
    cudaMalloc((void**)&un_sign_as, sizeof(int));
    cudaMemset(un_sign_as, 0, sizeof(int));

    int *un_sign_ds;
    cudaMalloc((void**)&un_sign_ds, sizeof(int));
    cudaMemset(un_sign_ds, 0, sizeof(int));

    
    
    std::vector<int> h_all_p_max(num1);
    std::vector<int> h_all_p_min(num1);


    cudaMemcpyToSymbol(width, &width1, sizeof(int), 0, cudaMemcpyHostToDevice);
    
    cudaMemcpyToSymbol(height, &height1, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(depth, &depth1, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(num, &num1, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(bound, &bound1, sizeof(double), 0, cudaMemcpyHostToDevice);
    
    cudaStatus = cudaMalloc(&temp, num1 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaStatus = cudaMalloc(&temp1, num1 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaStatus = cudaMalloc(&temp3, num1  * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }

    cudaStatus = cudaMalloc(&temp4, num1  * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    
    cudaStatus = cudaMalloc(&d_data, num1 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaStatus = cudaMalloc(&or_l, num1 * 2  * sizeof(int));
    cudaMalloc(&dec_l, num1 * 2 * sizeof(int));
    
    
    cudaEvent_t start, stop;

    cudaEventCreate(&start);

    cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    

    cudaMemcpy(temp3, input_data1->data(), num1 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(temp4, decp_data1->data(), num1 * sizeof(double), cudaMemcpyHostToDevice);
    
    change_mode<<<1,1>>>(neighbor_number);
    cudaDeviceSynchronize();
    

    int *d_temp;  
    size_t size = num1 * sizeof(int);

    cudaStatus = cudaMalloc(&d_temp, size);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }

    cudaMemcpyToSymbol(all_max, &d_temp, sizeof(int*));
    

    
    int *d_temp1;  
    size_t size1 = num1 * sizeof(int);
    cudaStatus = cudaMalloc(&d_temp1, size1);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaMemcpyToSymbol(all_min, &d_temp1, sizeof(int*));

    int *p_temp; 
    cudaStatus = cudaMalloc(&p_temp, size1);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaMemcpyToSymbol(all_p_min, &p_temp, sizeof(int*));

    int *p_temp1;  
    
    cudaStatus = cudaMalloc(&p_temp1, size1);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaMemcpyToSymbol(all_p_max, &p_temp1, sizeof(int*));

    int *d_temp2;  
    size_t size4 = num1  * sizeof(int);
    
    cudaStatus = cudaMalloc(&d_temp2, size4);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaMemcpyToSymbol(de_direction_as, &d_temp2, sizeof(int*));
    cudaMemcpyToSymbol(or_label, &or_l, sizeof(int*));
    cudaMemcpyToSymbol(dec_label, &dec_l, sizeof(int*));

    int *d_temp3;  
    size_t size3 = num1 * sizeof(int);
    cudaStatus = cudaMalloc(&d_temp3, size3);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    
    cudaMemcpyToSymbol(de_direction_ds, &d_temp3, sizeof(int*));
    cudaMemcpyToSymbol(or_maxi, &temp, sizeof(int*));
    cudaMemcpyToSymbol(or_mini, &temp1, sizeof(int*));
    cudaMemcpyToSymbol(input_data, &temp3, sizeof(double*));
    cudaMemcpyToSymbol(decp_data, &temp4, sizeof(double*));

    
    
    dim3 blockSize(256);
    dim3 gridSize((num1 + blockSize.x - 1) / blockSize.x);
    
    int* tempDevicePtr = nullptr;
    size_t arraySize = num1*max_n; 
    cudaStatus = cudaMalloc(&tempDevicePtr, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaMemcpyToSymbol(adjacency, &tempDevicePtr, sizeof(tempDevicePtr));
   

    computeAdjacency<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();
    find_direction<<<gridSize, blockSize>>>(1);
    
    
    
   
    double init_value = -2*bound1;
    double* buffer_temp;
    cudaStatus = cudaMalloc(&buffer_temp, num1  * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaMemcpyToSymbol(d_deltaBuffer, &buffer_temp, sizeof(double*));

    double* array_temp;
    cudaStatus = cudaMalloc(&array_temp, num1  * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaMemcpyToSymbol(id_array, &array_temp, sizeof(int*));
   
    find_direction<<<gridSize, blockSize>>>();
   
    
    int initialValue = 0;
    cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
    iscriticle<<<gridSize,blockSize>>>();
    
    
    // double h_s[num1];
    int host_count_f_max;
    cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
    
    int host_count_f_min;
    cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
    
    
    int cnt  = 0;
    
    std::vector<int> h_all_max(num1);

    if(preserve_max == 0) host_count_f_max = 0;
    if(preserve_min == 0) host_count_f_min = 0;
    
    
    while(host_count_f_min>0 || host_count_f_max>0){
            
            
            initializeKernel<<<gridSize, blockSize>>>(init_value);
            
            cudaDeviceSynchronize();
            
            dim3 blockSize1(256);
            dim3 gridSize1((host_count_f_max + blockSize1.x - 1) / blockSize1.x);
            
            if(preserve_max == 1)
            {
                fix_maxi_critical<<<gridSize1, blockSize1>>>(0,cnt);
            }   
 
            dim3 blocknum(256);
            dim3 gridnum((host_count_f_min + blocknum.x - 1) / blocknum.x);
            if(preserve_min == 1)
            {
                fix_maxi_critical<<<gridnum, blocknum>>>(1,cnt);
            }
            
            applyDeltaBuffer<<<gridSize, blockSize>>>();
            cudaDeviceSynchronize();
            cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
            cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
            
            cudaDeviceSynchronize();
            
            iscriticle<<<gridSize, blockSize>>>();
            find_direction<<<gridSize,blockSize>>>();

            cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
            cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
            
            cudaDeviceSynchronize();
            if(preserve_max == 0) host_count_f_max = 0;
            if(preserve_min == 0) host_count_f_min = 0;
            
    }

    
    if(preserve_path ==0 || preserve_max == 0 || preserve_min == 0) 
    {
        cudaMemcpy(decp_data1->data(), temp4, num1 * sizeof(double), cudaMemcpyDeviceToHost);
        return MSZ_ERR_NO_ERROR;
    }
    

    initializeWithIndex<<<gridSize, blockSize>>>(num1,0);
    initializeWithIndex<<<gridSize, blockSize>>>(num1,1);
    
   
    while(h_un_sign_as>0 or h_un_sign_ds>0){
        
        int zero = 0;
        int zero1 = 0;

        
        cudaMemcpy(un_sign_as, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(un_sign_ds, &zero1, sizeof(int), cudaMemcpyHostToDevice);
        getlabel<<<gridSize, blockSize>>>(un_sign_as,un_sign_ds,0);
        
        cudaMemcpy(&h_un_sign_as, un_sign_as, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_un_sign_ds, un_sign_ds, sizeof(int), cudaMemcpyDeviceToHost);
        
        
    }   
    
    
    
    
    
    h_un_sign_as = num1;
    h_un_sign_ds = num1;
    while(h_un_sign_as>0 or h_un_sign_ds>0){
        
        int zero = 0;
        int zero1 = 0;


        cudaMemcpy(un_sign_as, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(un_sign_ds, &zero1, sizeof(int), cudaMemcpyHostToDevice);
        getlabel<<<gridSize, blockSize>>>(un_sign_as,un_sign_ds,1);
        
        cudaMemcpy(&h_un_sign_as, un_sign_as, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_un_sign_ds, un_sign_ds, sizeof(int), cudaMemcpyDeviceToHost);
        
        
        
    }
    
    
    cudaMemcpy(dec_label1->data(), dec_l, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(or_label1->data(), or_l, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    
    
    
    cudaMemcpyToSymbol(count_p_max, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(count_p_min, &initialValue, sizeof(int));
    get_wrong_index_path<<<gridSize, blockSize>>>();

    int host_count_p_max;
    
    cudaMemcpyFromSymbol(&host_count_p_max, count_p_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
    
    int host_count_p_min;
    cudaMemcpyFromSymbol(&host_count_p_min, count_p_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
    
    
    while(host_count_p_min>0 or host_count_p_max>0 or host_count_f_min>0 or host_count_f_max>0){
        
     

        initializeKernel<<<gridSize, blockSize>>>(init_value);
        dim3 blockSize2(256);
        dim3 gridSize2((host_count_p_max + blockSize2.x - 1) / blockSize2.x);


        
        fixpath<<<gridSize2, blockSize2>>>(0);
        cudaDeviceSynchronize();
        
        
        
        dim3 blockSize3(256);
        dim3 gridSize3((host_count_p_min + blockSize3.x - 1) / blockSize3.x);
        fixpath<<<gridSize3, blockSize3>>>(1);
        cudaDeviceSynchronize();

        applyDeltaBuffer<<<gridSize, blockSize>>>();
        cudaDeviceSynchronize();
        
        find_direction<<<gridSize, blockSize>>>();
        cudaDeviceSynchronize();

        

        cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
        
        iscriticle<<<gridSize, blockSize>>>();
        cudaDeviceSynchronize();

        cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
        
        while(host_count_f_max>0 or host_count_f_min>0){
            dim3 blockSize1(256);
            dim3 gridSize1((host_count_f_max + blockSize1.x - 1) / blockSize1.x);
            
            initializeKernel<<<gridSize, blockSize>>>(init_value);
            
            fix_maxi_critical<<<gridSize1, blockSize1>>>(0,cnt);
            
            cudaDeviceSynchronize();
            // cudaDeviceSynchronize();
            
            
            dim3 blocknum(256);
            dim3 gridnum((host_count_f_min + blocknum.x - 1) / blocknum.x);
            
            fix_maxi_critical<<<gridnum, blocknum>>>(1,cnt);
            cudaDeviceSynchronize();
            
            
            
            cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
            cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
            
            applyDeltaBuffer<<<gridSize, blockSize>>>();
            find_direction<<<gridSize,blockSize>>>();
            iscriticle<<<gridSize, blockSize>>>();
            
            cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
            
            cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
            
            cudaDeviceSynchronize();
            
           
        }
        
        initializeWithIndex<<<gridSize, blockSize>>>(num1,0);
        
        h_un_sign_as = num1;
        h_un_sign_ds = num1;
        
        while(h_un_sign_as>0 or h_un_sign_ds>0){
        
            int zero = 0;
            int zero1 = 0;

            
            cudaMemcpy(un_sign_as, &zero, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(un_sign_ds, &zero1, sizeof(int), cudaMemcpyHostToDevice);
            getlabel<<<gridSize, blockSize>>>(un_sign_as,un_sign_ds,0);
            
            cudaMemcpy(&h_un_sign_as, un_sign_as, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_un_sign_ds, un_sign_ds, sizeof(int), cudaMemcpyDeviceToHost);
           
            
            
        } 
        
        
        cudaMemcpyToSymbol(count_p_max, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(count_p_min, &initialValue, sizeof(int));

        
        get_wrong_index_path<<<gridSize, blockSize>>>();
       

        cudaMemcpyFromSymbol(&host_count_p_max, count_p_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
        
        cudaMemcpyFromSymbol(&host_count_p_min, count_p_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
        
        cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
        cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));

        
        iscriticle<<<gridSize, blockSize>>>();
        


        cudaMemcpyFromSymbol(&host_count_f_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
       
        cudaMemcpyFromSymbol(&host_count_f_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
        
    
    }
    
    
    
    
    cudaMemcpy(a->data(), temp, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(b->data(), temp1, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(c->data(), d_temp2, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(d->data(), d_temp3, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(decp_data1->data(), temp4, num1 * sizeof(double), cudaMemcpyDeviceToHost);
    

    return MSZ_ERR_NO_ERROR;
}

int count_false_cases(std::vector<int> *a,std::vector<int> *b,
                std::vector<int> *c,std::vector<int> *d,
                std::vector<double> *input_data1,std::vector<double> *decp_data1,
                std::vector<int>* dec_label1,std::vector<int>* or_label1, 
                int width1, int height1, int depth1, int neighbor_number,
                int &wrong_min, int &wrong_max,  int &wrong_labels, int device_id)
{
    cudaError_t cudaStatus = cudaSetDevice(device_id);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_NO_AVAILABLE_GPU;
    }
    int *temp, *temp1, *d_data, *or_l, *dec_l;
    double *temp3, *temp4;
    int max_n = 12;
    if(neighbor_number!=0) max_n = 26;
    int num1 = width1*height1*depth1;
    
    int h_un_sign_as = num1;
    int h_un_sign_ds = num1;

    
    int *un_sign_as;
    cudaMalloc((void**)&un_sign_as, sizeof(int));
    cudaMemset(un_sign_as, 0, sizeof(int));

    int *un_sign_ds;
    cudaMalloc((void**)&un_sign_ds, sizeof(int));
    cudaMemset(un_sign_ds, 0, sizeof(int));


    cudaMemcpyToSymbol(width, &width1, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(height, &height1, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(depth, &depth1, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(num, &num1, sizeof(int), 0, cudaMemcpyHostToDevice);
    
    cudaStatus = cudaMalloc(&temp, num1 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaStatus = cudaMalloc(&temp1, num1 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaStatus = cudaStatus = cudaMalloc(&temp3, num1  * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaStatus = cudaMalloc(&temp4, num1  * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    
    cudaStatus = cudaMalloc(&d_data, num1 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaStatus = cudaMalloc(&or_l, num1 * 2  * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaStatus = cudaMalloc(&dec_l, num1 * 2 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    

    

    cudaStatus = cudaMemcpy(temp3, input_data1->data(), num1 * sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(temp4, decp_data1->data(), num1 * sizeof(double), cudaMemcpyHostToDevice);
    
    
    cudaDeviceSynchronize();
    change_mode<<<1,1>>>(neighbor_number);
    cudaDeviceSynchronize();
    

    
    int *d_temp;  
    size_t size = num1 * sizeof(int);

    
    cudaStatus = cudaMalloc(&d_temp, size);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }

    cudaMemcpyToSymbol(all_max, &d_temp, sizeof(int*));
    

    
    int *d_temp1;  
    size_t size1 = num1 * sizeof(int);

    
    cudaStatus = cudaMalloc(&d_temp1, size1);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaMemcpyToSymbol(all_min, &d_temp1, sizeof(int*));

    int *p_temp; 
    cudaStatus = cudaMalloc(&p_temp, size1);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaMemcpyToSymbol(all_p_min, &p_temp, sizeof(int*));

    int *p_temp1;  
    cudaStatus = cudaMalloc(&p_temp1, size1);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaMemcpyToSymbol(all_p_max, &p_temp1, sizeof(int*));

    int *d_temp2;  
    size_t size4 = num1  * sizeof(int);
    
    cudaStatus = cudaMalloc(&d_temp2, size4);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaMemcpyToSymbol(de_direction_as, &d_temp2, sizeof(int*));
    cudaMemcpyToSymbol(or_label, &or_l, sizeof(int*));
    cudaMemcpyToSymbol(dec_label, &dec_l, sizeof(int*));

    int *d_temp3;  
    size_t size3 = num1 * sizeof(int);

    
    cudaStatus = cudaMalloc(&d_temp3, size3);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    
    cudaMemcpyToSymbol(de_direction_ds, &d_temp3, sizeof(int*));
    cudaMemcpyToSymbol(or_maxi, &temp, sizeof(int*));
    cudaMemcpyToSymbol(or_mini, &temp1, sizeof(int*));
    cudaMemcpyToSymbol(input_data, &temp3, sizeof(double*));
    cudaMemcpyToSymbol(decp_data, &temp4, sizeof(double*));

    
    
    dim3 blockSize(256);
    dim3 gridSize((num1 + blockSize.x - 1) / blockSize.x);
    int* tempDevicePtr = nullptr;
    size_t arraySize = num1*max_n; 
    cudaStatus = cudaMalloc(&tempDevicePtr, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaMemcpyToSymbol(adjacency, &tempDevicePtr, sizeof(tempDevicePtr));

    computeAdjacency<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();

    find_direction<<<gridSize, blockSize>>>(1);
    
    double* buffer_temp;
    cudaStatus = cudaMalloc(&buffer_temp, num1  * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaMemcpyToSymbol(d_deltaBuffer, &buffer_temp, sizeof(double*));

    double* array_temp;
    cudaStatus = cudaMalloc(&array_temp, num1  * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return MSZ_ERR_OUT_OF_MEMORY;
    }
    cudaMemcpyToSymbol(id_array, &array_temp, sizeof(int*));
   
    find_direction<<<gridSize, blockSize>>>();
    
   
    
    int initialValue = 0;
    cudaMemcpyToSymbol(count_f_max, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(count_f_min, &initialValue, sizeof(int));
    iscriticle<<<gridSize,blockSize>>>();
    
    
    
    cudaMemcpyFromSymbol(&wrong_max, count_f_max, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&wrong_min, count_f_min, sizeof(int), 0, cudaMemcpyDeviceToHost);
    

    initializeWithIndex<<<gridSize, blockSize>>>(num1,0);
    initializeWithIndex<<<gridSize, blockSize>>>(num1,1);
   
    while(h_un_sign_as>0 or h_un_sign_ds>0){
        
        int zero = 0;
        int zero1 = 0;

        cudaMemcpy(un_sign_as, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(un_sign_ds, &zero1, sizeof(int), cudaMemcpyHostToDevice);
        getlabel<<<gridSize, blockSize>>>(un_sign_as,un_sign_ds,0);
        
        cudaMemcpy(&h_un_sign_as, un_sign_as, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_un_sign_ds, un_sign_ds, sizeof(int), cudaMemcpyDeviceToHost);
        
        
    }   
    
    h_un_sign_as = num1;
    h_un_sign_ds = num1;
    while(h_un_sign_as>0 or h_un_sign_ds>0){
        
        int zero = 0;
        int zero1 = 0;


        cudaMemcpy(un_sign_as, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(un_sign_ds, &zero1, sizeof(int), cudaMemcpyHostToDevice);
        getlabel<<<gridSize, blockSize>>>(un_sign_as,un_sign_ds,1);
        
        cudaMemcpy(&h_un_sign_as, un_sign_as, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_un_sign_ds, un_sign_ds, sizeof(int), cudaMemcpyDeviceToHost);
      
    }

    cudaMemcpy(dec_label1->data(), dec_l, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(or_label1->data(), or_l, num1 * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaMemcpyToSymbol(count_p_max, &initialValue, sizeof(int));
    cudaMemcpyToSymbol(count_p_min, &initialValue, sizeof(int));
    get_wrong_index_count<<<gridSize, blockSize>>>();

    
    cudaMemcpyFromSymbol(&wrong_labels, count_false_label, sizeof(int), 0, cudaMemcpyDeviceToHost);
    
    
    return MSZ_ERR_NO_ERROR;
}
