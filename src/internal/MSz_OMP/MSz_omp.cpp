#include "MSz_omp.h"
#include "MSz_globals.h"
#include <parallel/algorithm>  
#include <omp.h>
#include <ostream>
#include <iostream>

void computeAdjacency(std::vector<int>& adjacency, int width, int height, int depth, int maxNeighbors) {
    int data_size = width * height * depth;
    #pragma omp parallel for
    for(int i=0;i<data_size;i++){
        int y = (i / (width)) % height; // Get the x coordinate
        int x = i % width; // Get the y coordinate
        int z = (i / (width * height)) % depth;
        int neighborIdx = 0;
        for (int d = 0; d < maxNeighbors; d++) {
            
            int dirX = directions[d * 3];     
            int dirY = directions[d * 3 + 1]; 
            int dirZ = directions[d * 3 + 2]; 
            int newX = x + dirX;
            int newY = y + dirY;
            int newZ = z + dirZ;
            int r = newX + newY * width + newZ* (height * width); // Calculate the index of the adjacent vertex
            
            if (newX >= 0 && newX < width && newY >= 0 && newY < height && r < width*height*depth && newZ<depth && newZ>=0) {
                
                adjacency[i * maxNeighbors + neighborIdx] = r;
                neighborIdx++;

            }
        // Fill the remaining slots with -1 or another placeholder value
        for (int j = neighborIdx; j < maxNeighbors; ++j) {
            adjacency[i * maxNeighbors + j] = -1;
        }
    }
    
    }
}

int find_direction(const std::vector<double>* data, const std::vector<int>* adjacency, std::vector<int>& direction_as, std::vector<int>& direction_ds, int width, int height, int depth, int maxNeighbors){
    int data_size = width * height * depth;
    #pragma omp parallel for
    
    for (int index=0;index<data_size;index++){
        
        int largetst_index = index;
        for(int j =0;j<maxNeighbors;++j){
            int i = (*adjacency)[index*maxNeighbors+j];
            if(i!=-1 and ((*data)[i]>(*data)[largetst_index] or ((*data)[i]==(*data)[largetst_index] and i>largetst_index))){
                
                largetst_index = i;
            };
        };
        int x_diff = (largetst_index % width) - (index % width);
        int y_diff = (largetst_index / (width)) % height - (index / (width)) % height;
        int z_diff = (largetst_index /(width * height)) % depth - (index /(width * height)) % depth;
        direction_as[index] = getDirection(x_diff, y_diff,z_diff, maxNeighbors);
        
        
        largetst_index = index;
        for(int j =0;j<maxNeighbors;++j){
            int i = (*adjacency)[index*maxNeighbors+j];
            
            if(i!=-1 and ((*data)[i]<(*data)[largetst_index] or ((*data)[i]==(*data)[largetst_index] and i<largetst_index))){
                largetst_index = i;  
            };
        };
        
        
    
        y_diff = (largetst_index / (width)) % height - (index / (width)) % height;
        x_diff = (largetst_index % width) - (index % width);
        z_diff = (largetst_index /(width * height)) % depth - (index /(width * height)) % depth;
        int t = getDirection(x_diff, y_diff,z_diff, maxNeighbors);  
        direction_ds[index] = getDirection(x_diff, y_diff,z_diff, maxNeighbors);     

        
    };
    return 0;
};

void initializeWithIndex(std::vector<int>& label, const std::vector<int>* direction_ds, const std::vector<int>* direction_as) {
    int data_size = label.size()/2;
    #pragma omp parallel for
    for(int index=0;index<data_size;index++){
        
        if((*direction_ds)[index]!=-1){
            label[index*2] = index;
        }
        else{
            label[index*2] = -1;
        }

        if((*direction_as)[index]!=-1){
            label[index*2+1] = index;
        }
        else{
            label[index*2+1] = -1;
        }
    }
};

void getlabel(std::vector<int>& label, const std::vector<int>* direction_as, const std::vector<int>* direction_ds, int& un_sign_ds, int& un_sign_as,
            int width, int height, int depth, int maxNeighbors
){
    
    un_sign_ds = 0;
    un_sign_as = 0;
    int data_size = label.size()/2;
    #pragma omp parallel for reduction(+:un_sign_as) reduction(+:un_sign_ds)
    
    for(int i=0;i<data_size;i++){
        int cur = label[i*2+1];
        int next_vertex;
        if (cur!=-1 and (*direction_as)[cur]!=-1){
            
            int direc = (*direction_as)[cur];
            next_vertex = from_direction_to_index(cur, direc, width, height, depth, maxNeighbors);
            label[i*2+1] = next_vertex;
            if ((*direction_as)[label[i*2+1]] != -1){
                un_sign_as+=1;  
            }
            
        }

    
        cur = label[i*2];
        int next_vertex1;
        
        if (cur!=-1 and label[cur*2]!=-1){
            
            int direc = (*direction_ds)[cur];
            next_vertex1 = from_direction_to_index(cur, direc, width, height, depth, maxNeighbors);
            label[i*2] = next_vertex1;
            if ((*direction_ds)[label[i*2]]!=-1){
                un_sign_ds+=1;
                }
            } 
    }
    return;

}


void mappath(std::vector<int>& label, const std::vector<int> *direction_as, const std::vector<int> *direction_ds,
            int width, int height, int depth, int maxNeighbors
){

    int data_size = width * height * depth;
    int h_un_sign_as = data_size;
    int h_un_sign_ds = data_size;

    initializeWithIndex(label,direction_ds,direction_as);
    
    while(h_un_sign_as>0 or h_un_sign_ds>0){
        
        h_un_sign_as=0;
        h_un_sign_ds=0;
        
        getlabel(label,direction_as,direction_ds,h_un_sign_as,h_un_sign_ds, width, height, depth, maxNeighbors);
        
    }   

    return;
};

void get_false_criticle_points
    (std::atomic_int &count_f_max, std::atomic_int &count_f_min, const std::vector<int> *adjacency,
    const std::vector<double>* decp_data, const std::vector<int>* or_direction_as,
    const std::vector<int>* or_direction_ds, std::vector<int> &false_min, std::vector<int> &false_max,
    int maxNeighbors, int data_size

) {
    
    count_f_max=0;
    count_f_min=0;

    #pragma omp parallel for
    for (auto i = 0; i < data_size; i ++) {
            
            bool is_maxima = true;
            bool is_minima = true;
        
            for (int index=0; index<maxNeighbors; index++) {
                int j = (*adjacency)[i*maxNeighbors+index];
                if(j==-1){
                    continue;
                }
                if ((*decp_data)[j] > (*decp_data)[i]) {
                    
                    is_maxima = false;
                    
                    break;
                }
                else if((*decp_data)[j] == (*decp_data)[i] and j>i){
                    is_maxima = false;
                    break;
                }
            }

            for (int index=0;index< maxNeighbors;index++) {
                int j = (*adjacency)[i*maxNeighbors+index];
                if(j==-1){
                    continue;
                }
                
                if ((*decp_data)[j] < (*decp_data)[i]) {
                    is_minima = false;
                    break;
                }
                else if((*decp_data)[j] == (*decp_data)[i] and j<i){
                    is_minima = false;
                    break;
                }
            }
            
        
        if((is_maxima && (*or_direction_as)[i]!=-1) or (!is_maxima && (*or_direction_as)[i]==-1)){
            int idx_fp_max = std::atomic_fetch_add(&count_f_max, 1);
            false_max[idx_fp_max] = i;
            
        }
        
        else if ((is_minima && (*or_direction_ds)[i]!=-1) or (!is_minima && (*or_direction_ds)[i]==-1)) {
            
            int idx_fp_min = std::atomic_fetch_add(&count_f_min, 1);
            false_min[idx_fp_min] = i;
        }    
        
    }

}

void initialization(std::vector<double> &d_deltaBuffer, int data_size, double bound) 
{
    #pragma omp parallel for
    for(int i =0;i<data_size;i++){
        d_deltaBuffer[i] = - 4.0 * bound;
    }
}


void applyDeltaBuffer(const std::vector<double> *d_deltaBuffer, const std::vector<double> *input_data, std::vector<double> &decp_data, int data_size, double bound)
{
    #pragma omp parallel for
    
    for(int i=0;i<data_size;i++){
        
        if((*d_deltaBuffer)[i] > -4.0 * bound){
            
            if(std::abs((*d_deltaBuffer)[i]) > 1e-15) 
            {
                decp_data[i] += (*d_deltaBuffer)[i];
            }
            else decp_data[i] = (*input_data)[i] - bound;
        }
    } 
}

int fix_process_omp(std::vector<int> *or_direction_as,std::vector<int> *or_direction_ds,
            std::vector<int> *de_direction_as,std::vector<int> *de_direction_ds,
            const std::vector<double> *input_data, std::vector<double> *decp_data,
            std::vector<int>* dec_label,std::vector<int>* or_label, 
            int width, int height, int depth, 
            double bound, 
            int preserve_min, int preserve_max, 
            int preserve_path, int neighbor_number)
{
    
    int data_size = width * height * depth;

    std::vector<double> d_deltaBuffer;
    std::vector<int> adjacency, false_max, false_min;
    std::atomic<int>* id_array = new std::atomic<int>[data_size];
    int maxNeighbors = 26;

    d_deltaBuffer.resize(data_size,-4.0 * bound);

    adjacency.resize(data_size*maxNeighbors, -1);
    false_max.resize(data_size);
    false_min.resize(data_size);
    
    computeAdjacency(adjacency, width, height, depth, maxNeighbors);
    
    find_direction(input_data, &adjacency, *or_direction_as, *or_direction_ds, width, height, depth, maxNeighbors);
    find_direction(decp_data, &adjacency, *de_direction_as, *de_direction_ds, width, height, depth, maxNeighbors);

    initializeWithIndex(*or_label, or_direction_ds, or_direction_as);
    initializeWithIndex(*dec_label, de_direction_ds, de_direction_as);

    mappath(*or_label, or_direction_as, or_direction_ds, width, height, depth, maxNeighbors);
    

    std::atomic_int count_f_max = 0;
    std::atomic_int count_f_min = 0;
    
    get_false_criticle_points(count_f_max, count_f_min, &adjacency,
                                decp_data, or_direction_as, or_direction_ds, false_min, false_max, maxNeighbors, data_size);
    if(preserve_max == 0) count_f_max = 0;
    if(preserve_min == 0) count_f_min = 0;
    while (count_f_max>0 or count_f_min>0){
            
            initialization(d_deltaBuffer, data_size, bound);
            
            #pragma omp parallel for
            for(auto i = 0; i < count_f_max; i ++){
                
                int critical_i = false_max[i];
                
                fix_maxi_critical(input_data, decp_data, or_direction_as, or_direction_ds, 
                        de_direction_as, de_direction_ds, 
                        d_deltaBuffer,
                        width, height, depth, maxNeighbors, bound,
                        critical_i,0);

            }

            #pragma omp parallel for
            for(auto i = 0; i < count_f_min; i ++){

                int critical_i = false_min[i];
                fix_maxi_critical(input_data, decp_data, or_direction_as, or_direction_ds, 
                        de_direction_as, de_direction_ds, 
                        d_deltaBuffer,
                        width, height, depth, maxNeighbors, bound,
                        critical_i,1);

            }
                
            applyDeltaBuffer(&d_deltaBuffer, input_data, *decp_data, data_size, bound);
            
            get_false_criticle_points(count_f_max, count_f_min, &adjacency,
                                decp_data, or_direction_as, or_direction_ds, false_min, false_max, maxNeighbors, data_size);
            find_direction(decp_data, &adjacency, *de_direction_as, *de_direction_ds, width, height, depth, maxNeighbors);
            
            if(preserve_max == 0) count_f_max = 0;
            if(preserve_min == 0) count_f_min = 0;
    }

    if(preserve_path ==0 || preserve_max == 0 || preserve_min == 0) 
    {
        return MSZ_ERR_NO_ERROR;
    }
    mappath(*dec_label, de_direction_as, de_direction_ds, width, height, depth, maxNeighbors);
    
    
    
    std::vector<int> wrong_index_as;
    std::vector<int> wrong_index_ds;
    double ratio = get_wrong_index_path(or_label, dec_label, wrong_index_as, wrong_index_ds, data_size);
    
    while (wrong_index_as.size()>0 or wrong_index_ds.size()>0 or count_f_max>0 or count_f_min>0){
        
        initialization(d_deltaBuffer, data_size, bound);
        #pragma omp parallel for
        for(int i =0;i< wrong_index_as.size();i++){
            int j = wrong_index_as[i];
            
            fixpath(input_data, decp_data, or_direction_as, or_direction_ds, de_direction_as, de_direction_ds, 
                        d_deltaBuffer,j,0,id_array, bound, width, height, depth, maxNeighbors);
        };
        #pragma omp parallel for
        for(int i =0;i< wrong_index_ds.size();i++){
            int j = wrong_index_ds[i];
            
            fixpath(input_data, decp_data, or_direction_as, or_direction_ds, de_direction_as, de_direction_ds, 
                        d_deltaBuffer,j,1,id_array, bound, width, height, depth, maxNeighbors);
        };
        
        
        applyDeltaBuffer(&d_deltaBuffer, input_data, *decp_data, data_size, bound);
        find_direction(decp_data, &adjacency, *de_direction_as, *de_direction_ds, width, height, depth, maxNeighbors);
        get_false_criticle_points(count_f_max, count_f_min, &adjacency,
                                decp_data, or_direction_as, or_direction_ds, false_min, false_max, maxNeighbors, data_size);
        
        
        while (count_f_max>0 or count_f_min>0){
            
                initialization(d_deltaBuffer, data_size, bound);
                #pragma omp parallel for

                for(auto i = 0; i < count_f_max; i ++){
                    
                    int critical_i = false_max[i];
                    
                    fix_maxi_critical(input_data, decp_data, or_direction_as, or_direction_ds, 
                        de_direction_as, de_direction_ds, 
                        d_deltaBuffer,
                        width, height, depth, maxNeighbors, bound,
                        critical_i,0);

                }
                
                
                
                #pragma omp parallel for
                for(auto i = 0; i < count_f_min; i ++){

                    int critical_i = false_min[i];
                    fix_maxi_critical(input_data, decp_data, or_direction_as, or_direction_ds, 
                        de_direction_as, de_direction_ds, 
                        d_deltaBuffer,
                        width, height, depth, maxNeighbors, bound,
                        critical_i,1);

                }
                
                applyDeltaBuffer(&d_deltaBuffer, input_data, *decp_data, data_size, bound);
                find_direction(decp_data, &adjacency, *de_direction_as, *de_direction_ds, width, height, depth, maxNeighbors);
                get_false_criticle_points(count_f_max, count_f_min, &adjacency,
                decp_data, or_direction_as, or_direction_ds, false_min, false_max, maxNeighbors, data_size);
        }
        mappath(*dec_label, de_direction_as, de_direction_ds, width, height, depth, maxNeighbors);
        get_wrong_index_path(or_label, dec_label, wrong_index_as, wrong_index_ds, data_size);
    };
    return MSZ_ERR_NO_ERROR;
}

int count_false_cases_omp(std::vector<int> *or_direction_as,std::vector<int> *or_direction_ds,
            std::vector<int> *de_direction_as,std::vector<int> *de_direction_ds,
            const std::vector<double> *input_data, std::vector<double> *decp_data,
            std::vector<int>* dec_label,std::vector<int>* or_label, 
            int width, int height, int depth, 
            int neighbor_number,
            int &wrong_min, int &wrong_max,  int &wrong_labels)
{
    
    int data_size = width * height * depth;

    std::vector<double> d_deltaBuffer;
    std::vector<int> adjacency, false_max, false_min;
    std::atomic<int>* id_array = new std::atomic<int>[data_size];

    int maxNeighbors = neighbor_number == 1? 26:12;
    
    adjacency.resize(data_size*maxNeighbors, -1);
    false_max.resize(data_size);
    false_min.resize(data_size);
    
    computeAdjacency(adjacency, width, height, depth, maxNeighbors);
    
    find_direction(input_data, &adjacency, *or_direction_as, *or_direction_ds, width, height, depth, maxNeighbors);
    find_direction(decp_data, &adjacency, *de_direction_as, *de_direction_ds, width, height, depth, maxNeighbors);

    initializeWithIndex(*or_label, or_direction_ds, or_direction_as);
    initializeWithIndex(*dec_label, de_direction_ds, de_direction_as);

    mappath(*or_label, or_direction_as, or_direction_ds, width, height, depth, maxNeighbors);
    mappath(*dec_label, de_direction_as, de_direction_ds, width, height, depth, maxNeighbors);

    std::atomic_int count_f_max = 0;
    std::atomic_int count_f_min = 0;
    
    
    get_false_criticle_points(count_f_max, count_f_min, &adjacency,
                                decp_data, or_direction_as, or_direction_ds, false_min, false_max, maxNeighbors, data_size);
    
    
    wrong_labels = count_false_labels(or_label, dec_label, data_size);
    wrong_min = count_f_min;
    wrong_max = count_f_max;

    return MSZ_ERR_NO_ERROR;
}
    

