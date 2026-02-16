#ifndef MSZ_OMP_INTERNAL_H
#define MSZ_OMP_INTERNAL_H

#include <vector>
#include <cmath>
#include <atomic>
#include "api/MSz.h"


    
void computeAdjacency(std::vector<int>& adjacency, int width, int height, int depth, int maxNeighbors);

int getDirection(int x, int y, int z, int maxNeighbors);

int find_direction(const std::vector<double>* data, const std::vector<int>* adjacency, std::vector<int>& direction_as, 
                    std::vector<int>& direction_ds, int width, int height, 
                    int depth, int maxNeighbors);

void initializeWithIndex(std::vector<int>& label, const std::vector<int>* direction_ds, const std::vector<int>* direction_as);

void mappath(std::vector<int>& label, const std::vector<int> *direction_as, const std::vector<int> *direction_ds, int width, int height, int depth, int maxNeighbors);

void get_false_criticle_points(std::atomic_int &count_f_max, std::atomic_int &count_f_min, const std::vector<int> *adjacency,
                                const std::vector<double>* decp_data, const std::vector<int>* or_direction_as,
                                const std::vector<int>* or_direction_ds, std::vector<int> &false_min, std::vector<int> &false_max, int maxNeighbors, int data_size);

void initialization(std::vector<double> &d_deltaBuffer, int data_size, double bound);

int from_direction_to_index(int cur, int direc, int width, int height, int depth, int maxNeighbors);

void fix_maxi_critical(const std::vector<double> *input_data, const std::vector<double> *decp_data,
                        const std::vector<int> *or_direction_as, const std::vector<int> *or_direction_ds, 
                        const std::vector<int> *de_direction_as, const std::vector<int> *de_direction_ds, 
                        std::vector<double> &d_deltaBuffer,
                        int width, int height, int depth, int maxNeighbors, double bound,
                        int index, int direction);
//     bool atomicCASDouble(double* ptr, double old_val, double new_val);

//     int swap(int index, double delta, std::vector<double> &d_deltaBuffer);

void applyDeltaBuffer(const std::vector<double> *d_deltaBuffer, const std::vector<double> *input_data, std::vector<double> &decp_data,
                        int data_size, double bound);

int fixpath(const std::vector<double> *input_data, const std::vector<double> *decp_data,
                    const std::vector<int> *or_direction_as, const std::vector<int> *or_direction_ds, 
                    const std::vector<int> *de_direction_as, const std::vector<int> *de_direction_ds, 
                    std::vector<double> &d_deltaBuffer,int index, int direction, std::atomic<int>* id_array, double bound, int width, int height, int depth, int maxNeighbors);
double get_wrong_index_path(const std::vector<int> *or_label, const std::vector<int> *dec_label, std::vector<int> &wrong_index_as, std::vector<int> &wrong_index_ds, int data_size);
int fix_process_omp(std::vector<int> *or_direction_as,std::vector<int> *or_direction_ds,
        std::vector<int> *de_direction_as,std::vector<int> *de_direction_ds,
        const std::vector<double> *input_data, std::vector<double> *decp_data,
        std::vector<int>* dec_label,std::vector<int>* or_label, 
        int width, int height, int depth, 
        double bound, 
        int preserve_min, int preserve_max, 
        int preserve_path, int neighbor_number);

int count_false_cases_omp(std::vector<int> *or_direction_as,std::vector<int> *or_direction_ds,
            std::vector<int> *de_direction_as,std::vector<int> *de_direction_ds,
            const std::vector<double> *input_data, std::vector<double> *decp_data,
            std::vector<int>* dec_label,std::vector<int>* or_label, 
            int width, int height, int depth, 
            int neighbor_number,
            int &wrong_min, int &wrong_max, int &wrong_labels);

int extract_critical_points_omp(
        const std::vector<double> *data,
        std::vector<MSz_critical_point_t> &critical_points,
        unsigned int critical_point_types,
        int width, int height, int depth,
        int neighbor_number);

#endif // MSZ_OMP_INTERNAL_H
