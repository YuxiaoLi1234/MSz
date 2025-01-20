#include "../../include/api/MSz.h"
#include "./MSz_globals.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <fstream>
#include <sstream>
#include <cfloat>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <stdatomic.h>
#include <cmath>
#include <parallel/algorithm>  
#include <unordered_map>
#include <random>
#include <atomic>
#include <string>
#include <omp.h>
#include <iostream>
#include <unordered_set>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <utility>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <random>


extern "C"
{
    int directions[78] = {
        1, 0, 0, -1, 0, 0,   
        0, 1, 0, 0, -1, 0,
        -1, 1, 0,1, -1, 0, 
        0, 0, -1, 0, -1, 1,
        0, 0, 1, 0, 1, -1,
        -1, 0, 1, 1, 0, -1,
        1, 1, 0, -1, -1, 0,
        1, 0, 1, -1, 0, -1,
        0, 1, 1, 0, -1, -1,
        1, 1, 1, 1, 1, -1,  
        1, -1, 1, 1, -1, -1,
        -1, 1, 1, -1, 1, -1,
        -1, -1, 1, -1, -1, -1
    };

    int direction_to_index_mapping[26][3] = 
    {
        {1, 0, 0}, {-1, 0, 0},   
        {0, 1, 0}, {0, -1, 0},
        {-1, 1, 0}, {1, -1, 0}, 
        {0, 0, -1}, {0, -1, 1},
        {0, 0, 1}, {0, 1, -1},
        {-1, 0, 1}, {1, 0, -1},
        {1, 1, 0}, {-1, -1, 0},
        {1, 0, 1}, {-1, 0, -1},
        {0, 1, 1}, {0, -1, -1},
        {1, 1, 1}, {1, 1, -1},  
        {1, -1, 1}, {1, -1, -1},
        {-1, 1, 1}, {-1, 1, -1},
        {-1, -1, 1}, {-1, -1, -1}
    };

    int getDirection(int x, int y, int z, int maxNeighbors){
    
        for (int i = 0; i < maxNeighbors; ++i) {
            if (direction_to_index_mapping[i][0] == x && direction_to_index_mapping[i][1] == y && direction_to_index_mapping[i][2] == z) {
                return i+1;  
            }
        }
        return -1;  

    };


    int from_direction_to_index(int cur, int direc, int width, int height, int depth, int maxNeighbors){
    
        if (direc==-1) return cur;
        int x = cur % width;
        int y = (cur / width) % height;
        int z = (cur/(width * height))%depth;

        if (direc >= 1 && direc <= maxNeighbors) {
            int delta_row = direction_to_index_mapping[direc-1][0];
            int delta_col = direction_to_index_mapping[direc-1][1];
            int delta_dep = direction_to_index_mapping[direc-1][2];
            
            
            int next_row = x + delta_row;
            int next_col = y + delta_col;
            int next_dep = z + delta_dep;
            return next_row + next_col * width + next_dep * (height * width);
        }
        else {
            return -1;
        }
    };
    
    // bool atomicCASDouble(double* ptr, double old_val, double new_val) {
    //     bool swapped = false;
    //     {
    //         if (*ptr == old_val) {
    //             *ptr = new_val;
    //             swapped = true;
    //         }
    //     }
    //     return swapped;
    // }

    // while (update_successful==0) {
    //     double current_value = d_deltaBuffer[index];
    //     if (-delta > current_value) {
    //         double swapped = atomicCASDouble(&d_deltaBuffer[index], delta);
    //         if (swapped == current_value) {
    //             update_successful = 1;
                
    //         } else {
    //             double oldValue = d_deltaBuffer[index];
    //             oldValue = swapped;
    //         }
    //     } else {
    //         update_successful = 1; 
    //     }
    // }
    double atomicCASDouble(double* address, double expected, double desired) {
        // reinterpret_cast 将 double 转换为 uint64_t 以便原子操作
        uint64_t* address_as_ull = reinterpret_cast<uint64_t*>(address);
        uint64_t expected_ull = *reinterpret_cast<uint64_t*>(&expected);
        uint64_t desired_ull = *reinterpret_cast<uint64_t*>(&desired);

        // 使用 std::atomic_compare_exchange 强制类型转换
        std::atomic<uint64_t>* atomic_address = reinterpret_cast<std::atomic<uint64_t>*>(address_as_ull);
        atomic_address->compare_exchange_strong(expected_ull, desired_ull);

        // 返回操作前的值
        return *reinterpret_cast<double*>(&expected_ull);
    }

    void swap(int index, double delta, std::vector<double>& d_deltaBuffer) {
        int update_successful = 0;

        while (update_successful == 0) {
            double current_value = d_deltaBuffer[index];

            if (-delta > current_value) {
                
                double swapped = atomicCASDouble(&d_deltaBuffer[index], current_value, delta);
                if (swapped == current_value) {
                    update_successful = 1; 
                }
            } else {
                update_successful = 1; 
            }
        }
    }

    void fix_maxi_critical(const std::vector<double> *input_data, const std::vector<double> *decp_data,
                        const std::vector<int> *or_direction_as, const std::vector<int> *or_direction_ds, 
                        const std::vector<int> *de_direction_as, const std::vector<int> *de_direction_ds, 
                        std::vector<double> &d_deltaBuffer,
                        int width, int height, int depth, int maxNeighbors, double bound,
                        int index, int direction
    ){
            double delta;
            int next_vertex;
        
            
            if (direction == 0){
                
                if ((*or_direction_as)[index]!=-1){
                    next_vertex = from_direction_to_index(index,(*or_direction_as)[index], width, height, depth, maxNeighbors);
                    double d = (((*input_data)[index] - bound) + (*decp_data)[index]) / 2.0 - (*decp_data)[index];
                    if((*decp_data)[index]<(*decp_data)[next_vertex] or ((*decp_data)[index]==(*decp_data)[next_vertex] and index<next_vertex)){
                        return;
                    }
                    
                    double oldValue = d_deltaBuffer[index];
                    
                    if ( d > oldValue ) {
                        swap(index, d, d_deltaBuffer);
                    }  
                    return;
                
                }
                else{
                    
                    int largest_index = from_direction_to_index(index,(*de_direction_as)[index], width, height, depth, maxNeighbors);
                    
                    
                    if((*decp_data)[index]>(*decp_data)[largest_index] or((*decp_data)[index]==(*decp_data)[largest_index] and index>largest_index)){
                        return;
                    }
                    double d = (((*input_data)[largest_index] - bound) + (*decp_data)[largest_index]) / 2.0 - (*decp_data)[largest_index];
                    
                    double oldValue = d_deltaBuffer[largest_index];
                    if (d > oldValue) {
                        swap(largest_index, d, d_deltaBuffer);
                    }  

                    return;
                    
                    
                }
                
                
            
            }
            
            else if (direction != 0){
                
                
                if ((*or_direction_ds)[index]!=-1){
                    
                    next_vertex= from_direction_to_index(index,(*or_direction_ds)[index], width, height, depth, maxNeighbors);
                    
                    double d = (((*input_data)[next_vertex] - bound) + (*decp_data)[next_vertex]) / 2.0 - (*decp_data)[next_vertex];
                    
                    if((*decp_data)[index]>(*decp_data)[next_vertex] or ((*decp_data)[index]==(*decp_data)[next_vertex] and index>next_vertex)){
                        return;
                    }

                    double oldValue = d_deltaBuffer[next_vertex];
                    if (d > oldValue) {
                        swap(next_vertex, d, d_deltaBuffer);
                    }  

                    return;
                    
                
                }
            
                else{
                    
                    int largest_index = from_direction_to_index(index,(*de_direction_ds)[index], width, height, depth, maxNeighbors);
        
                    if((*decp_data)[index]<(*decp_data)[largest_index] or ((*decp_data)[index]==(*decp_data)[largest_index] and index<largest_index)){
                        
                        return;
                    }
                    
                    double d = (((*input_data)[index] - bound) + (*decp_data)[index]) / 2.0 - (*decp_data)[index];
                    
                    double oldValue = d_deltaBuffer[index];
                    if (d > oldValue) {
                        swap(index, d, d_deltaBuffer);
                    }  

                    return; 

                    
                }

                
            }    
            

            

            return;
    }

    double get_wrong_index_path(const std::vector<int> *or_label, const std::vector<int> *dec_label, std::vector<int> &wrong_index_as, std::vector<int> &wrong_index_ds, int data_size){
    
        wrong_index_as.clear();
        wrong_index_ds.clear();
        double cnt = 0.0;
        for (int index = 0; index < data_size; ++index) {
            
            if((*or_label)[index*2+1] != (*dec_label)[index*2+1] || (*or_label)[index*2] != (*dec_label)[index*2]){
                cnt+=1.0;
            }
            if ((*or_label)[index*2+1] != (*dec_label)[index*2+1]) {
                wrong_index_as.push_back(index);
            }
            if ((*or_label)[index*2] != (*dec_label)[index*2]) {
                wrong_index_ds.push_back(index);
            }
        }   
        double result = static_cast<double>(cnt) / static_cast<double>(data_size);
        return result;
    };

    double count_false_labels(const std::vector<int> *or_label, const std::vector<int> *dec_label, int data_size){
        int cnt = 0;
        for (int index = 0; index < data_size; ++index) {
            if((*or_label)[index*2+1] != (*dec_label)[index*2+1] || (*or_label)[index*2] != (*dec_label)[index*2]){
                cnt+=1;
            }
            
        }   
        return cnt;
    };

    int fixpath(const std::vector<double> *input_data, const std::vector<double> *decp_data,
                        const std::vector<int> *or_direction_as, const std::vector<int> *or_direction_ds, 
                        const std::vector<int> *de_direction_as, const std::vector<int> *de_direction_ds, 
                        std::vector<double> &d_deltaBuffer,int index, int direction, std::atomic<int>* id_array,
                        double bound, int width, int height, int depth, int maxNeighbors){
        double delta;
        if(direction == 0){
            int cur = index;
            while ((*or_direction_as)[cur] == (*de_direction_as)[cur]){
                int next_vertex =  from_direction_to_index(cur,(*de_direction_as)[cur], width, height, depth, maxNeighbors);
                
                if((*de_direction_as)[cur]==-1 && next_vertex == cur){
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
            
            
            if (start_vertex==-1) return 0;
            else{
                
                int false_index= from_direction_to_index(cur,(*de_direction_as)[cur], width, height, depth, maxNeighbors);
                int true_index= from_direction_to_index(cur, (*or_direction_as)[cur], width, height, depth, maxNeighbors);
                if(false_index==true_index) return 0;
                double d = (((*input_data)[false_index] - bound) + (*decp_data)[false_index]) / 2.0 - (*decp_data)[false_index];
                
                double oldValue = d_deltaBuffer[false_index];
                if (d > oldValue) {
                    swap(false_index, d, d_deltaBuffer);
                }  

                return 0;
            }
            
        }

        else 
        {
            
                
            
            int cur = index;
            
            
            while ((*or_direction_ds)[cur] == (*de_direction_ds)[cur]){
                
                int next_vertex = from_direction_to_index(cur,(*de_direction_ds)[cur], width, height, depth, maxNeighbors);
                if (next_vertex == cur){
                    cur = next_vertex;
                    break;
                }
                cur = next_vertex;
                    
            }
        
            int start_vertex = cur;
            if (start_vertex==-1) return 0;
            
            else{
                
                int false_index= from_direction_to_index(cur,(*de_direction_ds)[cur], width, height, depth, maxNeighbors);
                int true_index= from_direction_to_index(cur, (*or_direction_ds)[cur], width, height, depth, maxNeighbors);
                if(false_index==true_index) return 0;

                double d = (((*input_data)[true_index] - bound) + (*decp_data)[true_index]) / 2.0 - (*decp_data)[true_index];
                double oldValue = d_deltaBuffer[true_index];
                if (d > oldValue) {
                    swap(true_index, d, d_deltaBuffer);
                }  

                return 0;
            }
        
        }
        return 0;
    }


    
}
