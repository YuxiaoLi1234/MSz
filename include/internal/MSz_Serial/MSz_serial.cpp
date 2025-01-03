#include "../MSz_global/MSz_globals.h"
#include "../../api/MSz.h"
#include "MSz_serial.h"
// g++-12 -fopenmp -std=c++17 -O3 -g MSz_cpu.cpp -o MSz_cpu
// g++ -fopenmp hello2.o kernel.o -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64 -lcudart -o helloworld


extern "C"
{
    void computeAdjacency_cpu(std::vector<int>& adjacency, int width, int height, int depth, int maxNeighbors) {
        int data_size = width * height * depth;
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
            
            for (int j = neighborIdx; j < maxNeighbors; ++j) {
                adjacency[i * maxNeighbors + j] = -1;
            }
        }
        
        }
    }

    int find_direction_cpu(const std::vector<double>* data, const std::vector<int>* adjacency, std::vector<int>& direction_as, std::vector<int>& direction_ds, int width, int height, int depth, int maxNeighbors){
        int data_size = width * height * depth;

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

    void initializeWithIndex_cpu(std::vector<int>& label, const std::vector<int>* direction_ds, const std::vector<int>* direction_as) {
        int data_size = label.size()/2;
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

    void getlabel_cpu(std::vector<int>& label, const std::vector<int>* direction_as, const std::vector<int>* direction_ds, int& un_sign_ds_cpu, int& un_sign_as_cpu,
                int width, int height, int depth, int maxNeighbors
    ){
        
        un_sign_ds_cpu = 0;
        un_sign_as_cpu = 0;
        int data_size = label.size()/2;
        
        for(int i=0;i<data_size;i++){
            int cur = label[i*2+1];
            int next_vertex;
            if (cur!=-1 and (*direction_as)[cur]!=-1){
                
                int direc = (*direction_as)[cur];
                next_vertex = from_direction_to_index(cur, direc, width, height, depth, maxNeighbors);
                label[i*2+1] = next_vertex;
                if ((*direction_as)[label[i*2+1]] != -1){
                    un_sign_as_cpu+=1;  
                }
                
            }
 
        
            cur = label[i*2];
            int next_vertex1;
            
            if (cur!=-1 and label[cur*2]!=-1){
                
                int direc = (*direction_ds)[cur];
                next_vertex1 = from_direction_to_index(cur, direc, width, height, depth, maxNeighbors);
                label[i*2] = next_vertex1;
                if ((*direction_ds)[label[i*2]]!=-1){
                    un_sign_ds_cpu+=1;
                    }
                } 
        }
        return;

    }


    void mappath_cpu(std::vector<int>& label, const std::vector<int> *direction_as, const std::vector<int> *direction_ds,
                int width, int height, int depth, int maxNeighbors
    ){

        int data_size = width * height * depth;
        int h_un_sign_as_cpu = data_size;
        int h_un_sign_ds_cpu = data_size;
    
        initializeWithIndex_cpu(label,direction_ds,direction_as);
        
        while(h_un_sign_as_cpu>0 or h_un_sign_ds_cpu>0){
            
            h_un_sign_as_cpu=0;
            h_un_sign_ds_cpu=0;
            
            getlabel_cpu(label,direction_as,direction_ds,h_un_sign_as_cpu,h_un_sign_ds_cpu, width, height, depth, maxNeighbors);
            
        }   

        return;
    };

    void get_false_criticle_points_cpu
        (int &count_f_max, int &count_f_min, const std::vector<int> *adjacency,
        const std::vector<double>* decp_data, const std::vector<int>* or_direction_as,
        const std::vector<int>* or_direction_ds, std::vector<int> &false_min, std::vector<int> &false_max,
        int maxNeighbors, int data_size

    ) {
        
        count_f_max=0;
        count_f_min=0;
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
                
                false_max[count_f_max++] = i;
                
            }
            
            else if ((is_minima && (*or_direction_ds)[i]!=-1) or (!is_minima && (*or_direction_ds)[i]==-1)) {
                
                
                false_min[count_f_min++] = i;
            }    
            
        }

    }

    void initialization_cpu(std::vector<double> &d_deltaBuffer, int data_size, double bound) 
    {
        for(int i =0;i<data_size;i++){
            d_deltaBuffer[i] = -4.0 * bound;
        }
    }
    
    void applyDeltaBuffer_cpu(const std::vector<double> *d_deltaBuffer, const std::vector<double> *input_data, std::vector<double> &decp_data, int data_size, double bound) {
        
        for(int i=0;i<data_size;i++){
            if((*d_deltaBuffer)[i] > -4.0 * bound){
                if(std::abs((*d_deltaBuffer)[i]) > 1e-15) decp_data[i] += (*d_deltaBuffer)[i];
                else decp_data[i] = (*input_data)[i] - bound;
            }
        } 
    }

    int fix_process_cpu(std::vector<int> *or_direction_as,std::vector<int> *or_direction_ds,
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

        int maxNeighbors = neighbor_number == 1?26:12;
        d_deltaBuffer.resize(data_size,-2.0 * bound);
        adjacency.resize(data_size*maxNeighbors, -1);
        false_max.resize(data_size);
        false_min.resize(data_size);
        
        computeAdjacency_cpu(adjacency, width, height, depth, maxNeighbors);
        
        find_direction_cpu(input_data, &adjacency, *or_direction_as, *or_direction_ds, width, height, depth, maxNeighbors);
        find_direction_cpu(decp_data, &adjacency, *de_direction_as, *de_direction_ds, width, height, depth, maxNeighbors);

        initializeWithIndex_cpu(*or_label, or_direction_ds, or_direction_as);
        initializeWithIndex_cpu(*dec_label, de_direction_ds, de_direction_as);

        mappath_cpu(*or_label, or_direction_as, or_direction_ds, width, height, depth, maxNeighbors);
        

        int count_f_max = 0;
        int count_f_min = 0;
        
        get_false_criticle_points_cpu(count_f_max, count_f_min, &adjacency,
                                    decp_data, or_direction_as, or_direction_ds, false_min, false_max, maxNeighbors, data_size);
        
        if(preserve_max == 0) count_f_max = 0;
        if(preserve_min == 0) count_f_min = 0;

        while (count_f_max>0 or count_f_min>0){
                
                
                initialization_cpu(d_deltaBuffer, data_size, bound);

                for(auto i = 0; i < count_f_max; i ++){
                    
                    int critical_i = false_max[i];
                    
                    fix_maxi_critical(input_data, decp_data, or_direction_as, or_direction_ds, 
                            de_direction_as, de_direction_ds, 
                            d_deltaBuffer,
                            width, height, depth, maxNeighbors, bound,
                            critical_i,0);

                }
    
                for(auto i = 0; i < count_f_min; i ++){

                    int critical_i = false_min[i];
                    fix_maxi_critical(input_data, decp_data, or_direction_as, or_direction_ds, 
                            de_direction_as, de_direction_ds, 
                            d_deltaBuffer,
                            width, height, depth, maxNeighbors, bound,
                            critical_i,1);

                }
                    
                applyDeltaBuffer_cpu(&d_deltaBuffer, input_data, *decp_data, data_size, bound);
                find_direction_cpu(decp_data, &adjacency, *de_direction_as, *de_direction_ds, width, height, depth, maxNeighbors);
                get_false_criticle_points_cpu(count_f_max, count_f_min, &adjacency,
                                    decp_data, or_direction_as, or_direction_ds, false_min, false_max, maxNeighbors, data_size);
                if(preserve_max == 0) count_f_max = 0;
                if(preserve_min == 0) count_f_min = 0;
        }

        mappath_cpu(*dec_label, de_direction_as, de_direction_ds, width, height, depth, maxNeighbors);
        
        if(preserve_path ==0 || preserve_max == 0 || preserve_min == 0) return MSZ_ERR_NO_ERROR;
        
        std::vector<int> wrong_index_as;
        std::vector<int> wrong_index_ds;
        get_wrong_index_path(or_label, dec_label, wrong_index_as, wrong_index_ds, data_size);

        while (wrong_index_as.size()>0 or wrong_index_ds.size()>0 or count_f_max>0 or count_f_min>0){
            
            initialization_cpu(d_deltaBuffer, data_size, bound);
            
            for(int i =0;i< wrong_index_as.size();i++){
                int j = wrong_index_as[i];
                
                fixpath(input_data, decp_data, or_direction_as, or_direction_ds, de_direction_as, de_direction_ds, 
                            d_deltaBuffer,j,0,id_array, bound, width, height, depth, maxNeighbors);
            };
            
            for(int i =0;i< wrong_index_ds.size();i++){
                int j = wrong_index_ds[i];
                
                fixpath(input_data, decp_data, or_direction_as, or_direction_ds, de_direction_as, de_direction_ds, 
                            d_deltaBuffer,j,1,id_array, bound, width, height, depth, maxNeighbors);
            };
            
            
            applyDeltaBuffer_cpu(&d_deltaBuffer, input_data, *decp_data, data_size, bound);
            find_direction_cpu(decp_data, &adjacency, *de_direction_as, *de_direction_ds, width, height, depth, maxNeighbors);
            get_false_criticle_points_cpu(count_f_max, count_f_min, &adjacency,
                                    decp_data, or_direction_as, or_direction_ds, false_min, false_max, maxNeighbors, data_size);
            
            
            while (count_f_max>0 or count_f_min>0){
                
                    initialization_cpu(d_deltaBuffer, data_size, bound);
                    

                    for(auto i = 0; i < count_f_max; i ++){
                        
                        int critical_i = false_max[i];
                        
                        fix_maxi_critical(input_data, decp_data, or_direction_as, or_direction_ds, 
                            de_direction_as, de_direction_ds, 
                            d_deltaBuffer,
                            width, height, depth, maxNeighbors, bound,
                            critical_i,0);

                    }
                    
                    
                    for(auto i = 0; i < count_f_min; i ++){

                        int critical_i = false_min[i];
                        fix_maxi_critical(input_data, decp_data, or_direction_as, or_direction_ds, 
                            de_direction_as, de_direction_ds, 
                            d_deltaBuffer,
                            width, height, depth, maxNeighbors, bound,
                            critical_i,1);

                    }
                    
                    applyDeltaBuffer_cpu(&d_deltaBuffer, input_data, *decp_data, data_size, bound);
                    find_direction_cpu(decp_data, &adjacency, *de_direction_as, *de_direction_ds, width, height, depth, maxNeighbors);
                    get_false_criticle_points_cpu(count_f_max, count_f_min, &adjacency,
                    decp_data, or_direction_as, or_direction_ds, false_min, false_max, maxNeighbors, data_size);
            }
            mappath_cpu(*dec_label, de_direction_as, de_direction_ds, width, height, depth, maxNeighbors);
            get_wrong_index_path(or_label, dec_label, wrong_index_as, wrong_index_ds, data_size);
        };
        return MSZ_ERR_NO_ERROR;
    }

    int count_false_cases_cpu(std::vector<int> *or_direction_as,std::vector<int> *or_direction_ds,
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

        int maxNeighbors = neighbor_number == 1?26:12;
        
        adjacency.resize(data_size*maxNeighbors, -1);
        false_max.resize(data_size);
        false_min.resize(data_size);
        
        computeAdjacency_cpu(adjacency, width, height, depth, maxNeighbors);
        
        find_direction_cpu(input_data, &adjacency, *or_direction_as, *or_direction_ds, width, height, depth, maxNeighbors);
        find_direction_cpu(decp_data, &adjacency, *de_direction_as, *de_direction_ds, width, height, depth, maxNeighbors);

        initializeWithIndex_cpu(*or_label, or_direction_ds, or_direction_as);
        initializeWithIndex_cpu(*dec_label, de_direction_ds, de_direction_as);

        mappath_cpu(*or_label, or_direction_as, or_direction_ds, width, height, depth, maxNeighbors);
        mappath_cpu(*dec_label, de_direction_as, de_direction_ds, width, height, depth, maxNeighbors);

        int count_f_max = 0;
        int count_f_min = 0;
        
       
        get_false_criticle_points_cpu(count_f_max, count_f_min, &adjacency,
                                    decp_data, or_direction_as, or_direction_ds, false_min, false_max, maxNeighbors, data_size);
        
        
        wrong_labels = count_false_labels(or_label, dec_label, data_size);
        wrong_min = count_f_min;
        wrong_max = count_f_max;

        return MSZ_ERR_NO_ERROR;
    }
    
}